"""
Ridge Regression Encoder for fMRI → CLIP Mapping
================================================

Implements a reproducible Ridge baseline that maps preprocessed fMRI features
(after T0/T1/T2 pipeline) to CLIP 512D embeddings.

Scientific Design:
- Uses L2-regularized linear regression (Ridge) for stable parameter estimation
- L2-normalizes predictions to unit length for meaningful cosine similarity
- Hyperparameter selection on validation set only (no test leakage)
- Follows NSD best practices: train on split, validate, retrain on train+val, test once

References:
- Allen et al. (2022). "A massive 7T fMRI dataset to bridge cognitive neuroscience and AI"
- Ozcelik & VanRullen (2023). "Brain-optimized neural networks learn non-hierarchical models"
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import Ridge as SklearnRidge
    from sklearn.metrics import mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Ridge encoder will not work.")


class RidgeEncoder:
    """
    Ridge regression encoder for fMRI → CLIP embedding mapping.
    
    Maps preprocessed fMRI features (shape: n_voxels or k_pca) to CLIP embeddings (512D).
    Uses L2 regularization for stable parameter estimation with high-dimensional inputs.
    
    Scientific Rationale:
    - Ridge regularization prevents overfitting with high-dimensional fMRI (CVF Open Access)
    - L2-normalized outputs ensure cosine similarity is meaningful (standard in CLIP decoding)
    - Linear mapping is strong baseline: captures first-order relationships without overfitting
    
    Attributes:
        alpha: L2 regularization strength (higher = more regularization)
        model: sklearn Ridge regressor
        input_dim: Input feature dimension (set during fit)
        output_dim: Output dimension (always 512 for CLIP ViT-B/32)
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize Ridge encoder.
        
        Args:
            alpha: L2 regularization strength. Typical range: [0.1, 100].
                   Higher values = stronger regularization (smoother predictions).
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for Ridge encoder. Install: pip install scikit-learn")
        
        self.alpha = alpha
        self.model = None
        self.input_dim = None
        self.output_dim = 512  # CLIP ViT-B/32 embedding dimension
        
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fit Ridge regression from fMRI features to CLIP embeddings.
        
        Args:
            X: fMRI features, shape (n_samples, n_features)
               - Can be raw voxels, masked voxels, or PCA components
               - Should already be preprocessed (T0/T1/T2)
            Y: CLIP embeddings, shape (n_samples, 512)
               - Should be L2-normalized (standard CLIP output)
        
        Raises:
            ValueError: If shapes are incompatible or data is invalid
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Sample mismatch: X has {X.shape[0]} samples, Y has {Y.shape[0]}")
        
        if Y.shape[1] != self.output_dim:
            raise ValueError(f"Y must be shape (n_samples, 512), got {Y.shape}")
        
        if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
            raise ValueError("Input contains NaN values")
        
        self.input_dim = X.shape[1]
        
        # Fit Ridge regression
        # fit_intercept=True: learns bias term (accounts for mean offset)
        # solver="auto": sklearn chooses best solver for data size
        # random_state=42: reproducible initialization (matters for some solvers)
        self.model = SklearnRidge(
            alpha=self.alpha,
            fit_intercept=True,
            solver="auto",
            random_state=42
        )
        
        logger.info(f"Fitting Ridge encoder: {X.shape} → {Y.shape} (alpha={self.alpha:.3f})")
        self.model.fit(X, Y)
        logger.info(f"✅ Ridge fitted: {self.input_dim}D → {self.output_dim}D")
        
    def predict(self, X: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Predict CLIP embeddings from fMRI features.
        
        Args:
            X: fMRI features, shape (n_samples, n_features)
            normalize: If True, L2-normalize predictions to unit length
                       (required for cosine similarity and retrieval)
        
        Returns:
            Predicted CLIP embeddings, shape (n_samples, 512)
            If normalize=True, each row has L2 norm = 1.0
        
        Raises:
            ValueError: If model not fitted or input shape mismatch
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {X.shape[1]}")
        
        # Predict
        Y_pred = self.model.predict(X).astype(np.float32)
        
        # L2-normalize predictions for cosine similarity
        # Scientific justification: CLIP embeddings lie on unit hypersphere
        # Normalization ensures cosine(pred, target) is meaningful
        if normalize:
            norms = np.linalg.norm(Y_pred, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)  # Avoid division by zero
            Y_pred = Y_pred / norms
        
        return Y_pred
    
    def save(self, path: str) -> None:
        """
        Save Ridge encoder to disk.
        
        Args:
            path: Output path (will create parent directories if needed)
        """
        if self.model is None:
            raise ValueError("Cannot save unfitted model")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save both sklearn model and metadata
        state = {
            "model": self.model,
            "alpha": self.alpha,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"✅ Ridge encoder saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "RidgeEncoder":
        """
        Load Ridge encoder from disk.
        
        Args:
            path: Path to saved model
            
        Returns:
            Loaded RidgeEncoder instance
        """
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        # Reconstruct encoder
        encoder = cls(alpha=state["alpha"])
        encoder.model = state["model"]
        encoder.input_dim = state["input_dim"]
        encoder.output_dim = state["output_dim"]
        
        logger.info(f"✅ Ridge encoder loaded from {path} (alpha={encoder.alpha:.3f}, "
                   f"{encoder.input_dim}D → {encoder.output_dim}D)")
        
        return encoder
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """
        Get model weights for analysis/visualization.
        
        Returns:
            Dictionary with keys:
            - "coef": Coefficient matrix (n_features, 512)
            - "intercept": Bias vector (512,)
        """
        if self.model is None:
            raise ValueError("Model not fitted")
        
        return {
            "coef": self.model.coef_.T,  # Transpose to (n_features, 512)
            "intercept": self.model.intercept_,
        }


def evaluate_predictions(
    Y_true: np.ndarray, 
    Y_pred: np.ndarray,
    normalize: bool = True
) -> Dict[str, float]:
    """
    Evaluate prediction quality using standard metrics.
    
    Args:
        Y_true: Ground truth CLIP embeddings (n_samples, 512)
        Y_pred: Predicted CLIP embeddings (n_samples, 512)
        normalize: If True, L2-normalize before computing cosine
    
    Returns:
        Dictionary with metrics:
        - "cosine": Mean cosine similarity
        - "cosine_std": Std of cosine similarities
        - "mse": Mean squared error
    """
    # L2-normalize for cosine similarity (with safe division)
    if normalize:
        # Prevent divide-by-zero for degenerate rows
        Y_true_denom = np.linalg.norm(Y_true, axis=1, keepdims=True)
        Y_true_denom = np.maximum(Y_true_denom, 1e-8)
        Y_true_norm = Y_true / Y_true_denom
        
        Y_pred_denom = np.linalg.norm(Y_pred, axis=1, keepdims=True)
        Y_pred_denom = np.maximum(Y_pred_denom, 1e-8)
        Y_pred_norm = Y_pred / Y_pred_denom
    else:
        Y_true_norm = Y_true
        Y_pred_norm = Y_pred
    
    # Cosine similarity (per sample)
    cosines = np.sum(Y_true_norm * Y_pred_norm, axis=1)
    
    # MSE
    mse = mean_squared_error(Y_true, Y_pred)
    
    return {
        "cosine": float(np.mean(cosines)),
        "cosine_std": float(np.std(cosines)),
        "mse": float(mse),
    }
