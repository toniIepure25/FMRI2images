"""
MoCo-style Memory Queue for Contrastive Learning

Maintains a large queue of negative samples to address small batch size (batch=4) limitation.
Enables effective InfoNCE training with many negatives.
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MemoryQueue(nn.Module):
    """
    MoCo-style queue for storing negative samples in contrastive learning.
    
    Features:
    - FIFO queue of embeddings
    - GPU-efficient implementation
    - Optional ID storage for debugging
    - No gradient tracking for queue updates
    
    Args:
        queue_size: Maximum number of items in queue (e.g., 8192, 16384, 32768)
        embedding_dim: Dimension of embeddings to store
        store_ids: Whether to also store sample IDs (for debugging)
        
    Usage:
        queue = MemoryQueue(queue_size=8192, embedding_dim=768)
        
        # During training
        with torch.no_grad():
            queue.enqueue(gt_embeddings, ids=sample_ids)
        
        # Get negatives for contrastive loss
        negatives = queue.get_queue()  # (queue_size, embedding_dim)
    """
    
    def __init__(
        self,
        queue_size: int = 8192,
        embedding_dim: int = 768,
        store_ids: bool = False,
    ):
        super().__init__()
        
        self.queue_size = queue_size
        self.embedding_dim = embedding_dim
        self.store_ids = store_ids
        
        # Register buffers (not parameters, but part of state_dict)
        self.register_buffer("queue", torch.zeros(queue_size, embedding_dim))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_is_full", torch.zeros(1, dtype=torch.bool))
        
        if store_ids:
            self.register_buffer("queue_ids", torch.zeros(queue_size, dtype=torch.long))
        
        logger.info(f"Initialized MemoryQueue: size={queue_size}, dim={embedding_dim}")
    
    @torch.no_grad()
    def enqueue(
        self,
        embeddings: torch.Tensor,
        ids: Optional[torch.Tensor] = None,
    ):
        """
        Add embeddings to the queue (FIFO).
        
        Args:
            embeddings: (batch_size, embedding_dim) tensor
            ids: Optional (batch_size,) tensor of sample IDs
        """
        batch_size = embeddings.size(0)
        
        # Ensure embeddings are on same device
        embeddings = embeddings.to(self.queue.device)
        
        ptr = int(self.queue_ptr)
        
        # Handle wrap-around
        if ptr + batch_size > self.queue_size:
            # Split into two parts
            remaining = self.queue_size - ptr
            self.queue[ptr:] = embeddings[:remaining]
            self.queue[:batch_size - remaining] = embeddings[remaining:]
            
            if self.store_ids and ids is not None:
                ids = ids.to(self.queue_ids.device)
                self.queue_ids[ptr:] = ids[:remaining]
                self.queue_ids[:batch_size - remaining] = ids[remaining:]
            
            # Mark queue as full
            self.queue_is_full[0] = True
        else:
            self.queue[ptr:ptr + batch_size] = embeddings
            
            if self.store_ids and ids is not None:
                ids = ids.to(self.queue_ids.device)
                self.queue_ids[ptr:ptr + batch_size] = ids
            
            # Check if queue is now full
            if ptr + batch_size >= self.queue_size:
                self.queue_is_full[0] = True
        
        # Update pointer
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size
    
    def get_queue(self, detach: bool = True) -> torch.Tensor:
        """
        Get current queue contents.
        
        Args:
            detach: Whether to detach from computation graph (default: True)
            
        Returns:
            (queue_size, embedding_dim) or (current_size, embedding_dim) tensor
        """
        if self.queue_is_full:
            queue = self.queue
        else:
            # Queue not yet full, return only filled portion
            ptr = int(self.queue_ptr)
            queue = self.queue[:ptr]
        
        if detach:
            return queue.detach()
        return queue
    
    def get_ids(self) -> Optional[torch.Tensor]:
        """Get stored IDs if available."""
        if not self.store_ids:
            return None
        
        if self.queue_is_full:
            return self.queue_ids.clone()
        else:
            ptr = int(self.queue_ptr)
            return self.queue_ids[:ptr].clone()
    
    def __len__(self) -> int:
        """Current number of items in queue."""
        if self.queue_is_full:
            return self.queue_size
        return int(self.queue_ptr)
    
    def is_ready(self, min_size: int = 256) -> bool:
        """Check if queue has enough items for training."""
        return len(self) >= min_size
    
    def clear(self):
        """Clear the queue."""
        self.queue.zero_()
        self.queue_ptr.zero_()
        self.queue_is_full[0] = False
        if self.store_ids:
            self.queue_ids.zero_()
        logger.info("Queue cleared")
    
    def get_stats(self) -> dict:
        """Get queue statistics."""
        return {
            "queue_size": self.queue_size,
            "current_size": len(self),
            "is_full": bool(self.queue_is_full),
            "fill_percentage": len(self) / self.queue_size * 100,
            "ptr": int(self.queue_ptr),
        }


class DualQueue(nn.Module):
    """
    Dual queue system for storing both query and key embeddings.
    
    Useful for maintaining separate queues for predictions and targets,
    or for asymmetric contrastive learning setups.
    
    Args:
        queue_size: Size of each queue
        embedding_dim: Dimension of embeddings
        store_ids: Whether to store sample IDs
    """
    
    def __init__(
        self,
        queue_size: int = 8192,
        embedding_dim: int = 768,
        store_ids: bool = False,
    ):
        super().__init__()
        
        self.queue_query = MemoryQueue(queue_size, embedding_dim, store_ids)
        self.queue_key = MemoryQueue(queue_size, embedding_dim, store_ids)
        
        logger.info("Initialized DualQueue")
    
    @torch.no_grad()
    def enqueue(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        ids: Optional[torch.Tensor] = None,
    ):
        """Enqueue to both queues."""
        self.queue_query.enqueue(query_embeddings, ids)
        self.queue_key.enqueue(key_embeddings, ids)
    
    def get_queues(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get both queues."""
        return self.queue_query.get_queue(), self.queue_key.get_queue()
    
    def __len__(self) -> int:
        """Current size (assumes both queues same size)."""
        return len(self.queue_query)
    
    def is_ready(self, min_size: int = 256) -> bool:
        """Check if queues are ready."""
        return self.queue_query.is_ready(min_size) and self.queue_key.is_ready(min_size)
    
    def clear(self):
        """Clear both queues."""
        self.queue_query.clear()
        self.queue_key.clear()


# Factory function for easy instantiation
def create_memory_queue(
    config: dict,
    embedding_dim: int = 768,
) -> Optional[MemoryQueue]:
    """
    Create memory queue from config.
    
    Args:
        config: Dictionary with queue configuration:
            - enabled: bool
            - size: int
            - store_ids: bool
        embedding_dim: Dimension of embeddings
        
    Returns:
        MemoryQueue instance or None if disabled
    """
    if not config.get("enabled", False):
        logger.info("Memory queue disabled")
        return None
    
    queue = MemoryQueue(
        queue_size=config.get("size", 8192),
        embedding_dim=embedding_dim,
        store_ids=config.get("store_ids", False),
    )
    
    return queue
