"""
Embedding Geometry Preprocessing for CLIP Embeddings

Fixes anisotropy-driven chance identification by applying geometry normalization.
Implements two modes:
- center_pcr: subtract mean, remove top-k principal components, L2-normalize
- center_whiten: subtract mean, PCA-whiten, L2-normalize

Critical: fit on TRAIN ONLY, apply to val/test for proper evaluation.
"""

import numpy as np
import torch
import pickle
from pathlib import Path
from typing import Optional, Dict, Tuple, Literal
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PreprocessorArtifacts:
    """Stores fitted preprocessing artifacts."""
    mean: np.ndarray
    pca_components: Optional[np.ndarray] = None
    pca_explained_variance: Optional[np.ndarray] = None
    whitening_matrix: Optional[np.ndarray] = None
    mode: str = "center_pcr"
    k_components: Optional[int] = None
    
    def __post_init__(self):
        """Validate artifacts."""
        if self.mode == "center_pcr" and self.k_components is None:
            raise ValueError("center_pcr mode requires k_components")
        if self.mode == "center_whiten" and self.whitening_matrix is None:
            raise ValueError("center_whiten mode requires whitening_matrix")


class EmbeddingPreprocessor:
    """
    Embedding preprocessor for fixing anisotropic CLIP embedding spaces.
    
    Usage:
        # Fit on training data
        preprocessor = EmbeddingPreprocessor(mode="center_pcr", k_components=8)
        preprocessor.fit(train_embeddings)
        preprocessor.save("artifacts/preproc.pkl")
        
        # Apply to val/test
        val_embeddings_norm = preprocessor.transform(val_embeddings)
        
    Args:
        mode: "center_pcr" or "center_whiten"
        k_components: Number of top PCs to remove (for center_pcr)
        whiten_eps: Regularization for whitening (for center_whiten)
    """
    
    def __init__(
        self,
        mode: Literal["center_pcr", "center_whiten"] = "center_pcr",
        k_components: int = 8,
        whiten_eps: float = 1e-5,
        seed: int = 42,
    ):
        self.mode = mode
        self.k_components = k_components
        self.whiten_eps = whiten_eps
        self.seed = seed
        
        self.artifacts: Optional[PreprocessorArtifacts] = None
        self._is_fitted = False
        
    def fit(self, embeddings: np.ndarray) -> "EmbeddingPreprocessor":
        """
        Fit preprocessing on training embeddings.
        
        Args:
            embeddings: (N, D) array of embeddings
            
        Returns:
            self (for chaining)
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
        
        N, D = embeddings.shape
        logger.info(f"Fitting {self.mode} preprocessor on {N} embeddings (dim={D})")
        
        # Compute mean
        mean = embeddings.mean(axis=0)
        embeddings_centered = embeddings - mean
        
        if self.mode == "center_pcr":
            # PCA to remove top-k components
            cov = np.cov(embeddings_centered, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Sort descending
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Store top k components to remove
            pca_components = eigenvectors[:, :self.k_components]
            pca_explained_variance = eigenvalues[:self.k_components]
            
            total_var = eigenvalues.sum()
            removed_var = pca_explained_variance.sum()
            logger.info(
                f"Removing top {self.k_components} PCs: "
                f"{removed_var/total_var*100:.2f}% of variance"
            )
            
            self.artifacts = PreprocessorArtifacts(
                mean=mean,
                pca_components=pca_components,
                pca_explained_variance=pca_explained_variance,
                mode=self.mode,
                k_components=self.k_components,
            )
            
        elif self.mode == "center_whiten":
            # PCA whitening
            cov = np.cov(embeddings_centered, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Whitening matrix: V * Lambda^(-1/2) * V^T
            whitening_matrix = eigenvectors @ np.diag(
                1.0 / np.sqrt(eigenvalues + self.whiten_eps)
            ) @ eigenvectors.T
            
            logger.info(f"Built whitening matrix (shape={whitening_matrix.shape})")
            
            self.artifacts = PreprocessorArtifacts(
                mean=mean,
                whitening_matrix=whitening_matrix,
                mode=self.mode,
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        self._is_fitted = True
        return self
    
    def transform(
        self,
        embeddings: np.ndarray,
        return_torch: bool = False,
    ) -> np.ndarray:
        """
        Apply fitted preprocessing to embeddings.
        
        Args:
            embeddings: (N, D) array or (D,) vector
            return_torch: If True, return torch.Tensor instead of np.ndarray
            
        Returns:
            Normalized embeddings (same shape as input)
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        
        is_single = embeddings.ndim == 1
        if is_single:
            embeddings = embeddings[np.newaxis, :]
        
        # Center
        embeddings_proc = embeddings - self.artifacts.mean
        
        if self.mode == "center_pcr":
            # Remove top-k PCs
            projection = embeddings_proc @ self.artifacts.pca_components
            embeddings_proc = embeddings_proc - projection @ self.artifacts.pca_components.T
            
        elif self.mode == "center_whiten":
            # Whiten
            embeddings_proc = embeddings_proc @ self.artifacts.whitening_matrix
        
        # L2 normalize
        norms = np.linalg.norm(embeddings_proc, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        embeddings_proc = embeddings_proc / norms
        
        if is_single:
            embeddings_proc = embeddings_proc[0]
        
        if return_torch:
            return torch.from_numpy(embeddings_proc).float()
        return embeddings_proc
    
    def transform_torch(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Transform torch tensors (convenience wrapper).
        
        Args:
            embeddings: (N, D) or (D,) torch.Tensor
            
        Returns:
            Normalized embeddings (torch.Tensor)
        """
        device = embeddings.device
        dtype = embeddings.dtype
        
        embeddings_np = embeddings.cpu().numpy()
        embeddings_proc = self.transform(embeddings_np, return_torch=False)
        
        return torch.from_numpy(embeddings_proc).to(device=device, dtype=dtype)
    
    def save(self, path: Path | str) -> None:
        """Save fitted artifacts to disk."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted preprocessor")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump(self.artifacts, f)
        
        logger.info(f"Saved preprocessor artifacts to {path}")
    
    @classmethod
    def load(cls, path: Path | str) -> "EmbeddingPreprocessor":
        """Load fitted artifacts from disk."""
        path = Path(path)
        
        with open(path, "rb") as f:
            artifacts = pickle.load(f)
        
        instance = cls(
            mode=getattr(artifacts, "mode", "center_pcr"),
            k_components=getattr(artifacts, "pca_components", np.empty(0)).shape[0]
            if getattr(artifacts, "pca_components", None) is not None else 8,
        )
        instance.artifacts = artifacts
        instance._is_fitted = True
        logger.info(f"Loaded preprocessor artifacts from {path}")
        return instance
    
    def compute_diagnostics(
        self,
        embeddings: np.ndarray,
        n_pairs: int = 10000,
    ) -> Dict[str, float]:
        """
        Compute embedding space diagnostics.
        
        Args:
            embeddings: (N, D) embeddings to diagnose
            n_pairs: Number of random pairs to sample for anisotropy
            
        Returns:
            Dictionary of diagnostic metrics
        """
        N = len(embeddings)
        
        # Anisotropy score: mean cosine similarity between random pairs
        rng = np.random.RandomState(self.seed)
        idx1 = rng.randint(0, N, size=n_pairs)
        idx2 = rng.randint(0, N, size=n_pairs)
        
        # Ensure different pairs
        mask = idx1 != idx2
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        
        # Normalize for cosine
        emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        cosines = (emb_norm[idx1] * emb_norm[idx2]).sum(axis=1)
        
        anisotropy = cosines.mean()
        
        diagnostics = {
            "anisotropy_score": float(anisotropy),
            "anisotropy_std": float(cosines.std()),
            "num_embeddings": N,
            "embedding_dim": embeddings.shape[1],
        }
        
        # If fitted, compute before/after comparison
        if self._is_fitted:
            emb_transformed = self.transform(embeddings)
            emb_transformed_norm = emb_transformed / np.linalg.norm(
                emb_transformed, axis=1, keepdims=True
            )
            cosines_after = (emb_transformed_norm[idx1] * emb_transformed_norm[idx2]).sum(axis=1)
            
            diagnostics["anisotropy_score_after"] = float(cosines_after.mean())
            diagnostics["anisotropy_std_after"] = float(cosines_after.std())
            diagnostics["anisotropy_reduction"] = float(anisotropy - cosines_after.mean())
        
        return diagnostics


def compute_separation_histograms(
    embeddings: np.ndarray,
    labels: np.ndarray,
    preprocessor: Optional[EmbeddingPreprocessor] = None,
    n_pairs: int = 5000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute positive/negative cosine similarity histograms before/after preprocessing.
    
    Args:
        embeddings: (N, D) embeddings
        labels: (N,) integer labels for matching
        preprocessor: Optional fitted preprocessor
        n_pairs: Number of pairs to sample
        seed: Random seed
        
    Returns:
        pos_before, neg_before, pos_after, neg_after (each 1D array)
    """
    rng = np.random.RandomState(seed)
    N = len(embeddings)
    
    # Sample pairs
    idx1 = rng.randint(0, N, size=n_pairs)
    idx2 = rng.randint(0, N, size=n_pairs)
    mask = idx1 != idx2
    idx1 = idx1[mask]
    idx2 = idx2[mask]
    
    is_positive = labels[idx1] == labels[idx2]
    
    # Compute cosine similarities before
    emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    cosines_before = (emb_norm[idx1] * emb_norm[idx2]).sum(axis=1)
    
    pos_before = cosines_before[is_positive]
    neg_before = cosines_before[~is_positive]
    
    if preprocessor is not None and preprocessor._is_fitted:
        emb_transformed = preprocessor.transform(embeddings)
        emb_transformed_norm = emb_transformed / np.linalg.norm(
            emb_transformed, axis=1, keepdims=True
        )
        cosines_after = (emb_transformed_norm[idx1] * emb_transformed_norm[idx2]).sum(axis=1)
        
        pos_after = cosines_after[is_positive]
        neg_after = cosines_after[~is_positive]
    else:
        pos_after = np.array([])
        neg_after = np.array([])
    
    return pos_before, neg_before, pos_after, neg_after
