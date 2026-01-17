"""Contrastive learning components."""
from .queue import MemoryQueue, DualQueue, create_memory_queue

__all__ = ["MemoryQueue", "DualQueue", "create_memory_queue"]
