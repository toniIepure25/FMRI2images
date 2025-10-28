"""
Retrieval Evaluation Metrics
============================

Implements retrieval@K and cosine similarity for evaluating fMRI → CLIP decoding.

Scientific Context:
- Retrieval@K measures how often the true image appears in top-K predictions
- Standard metric in CLIP-based neural decoding (Ozcelik & VanRullen 2023)
- All vectors MUST be L2-normalized before scoring (CLIP embedding space convention)

References:
- Ozcelik & VanRullen (2023). "Brain-optimized neural networks"
- Radford et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision"
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity between two sets of vectors.
    
    IMPORTANT: Input vectors MUST be L2-normalized to unit length.
    If not normalized, this computes inner product, not cosine similarity.
    
    Args:
        a: Query vectors, shape (n_queries, d)
        b: Gallery vectors, shape (n_gallery, d)
        
    Returns:
        Similarity matrix, shape (n_queries, n_gallery)
        Values in [-1, 1] if inputs are normalized
    
    Example:
        >>> queries = np.random.randn(10, 512)
        >>> queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        >>> gallery = np.random.randn(100, 512)
        >>> gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
        >>> sim = cosine_sim(queries, gallery)  # (10, 100)
    """
    # Efficient matrix multiplication for cosine (when normalized)
    # sim[i, j] = a[i] · b[j] = cosine(a[i], b[j]) if ||a[i]|| = ||b[j]|| = 1
    return a @ b.T


def retrieval_at_k(
    query: np.ndarray,
    gallery: np.ndarray, 
    gt_index: np.ndarray,
    ks: Tuple[int, ...] = (1, 5, 10)
) -> Dict[str, float]:
    """
    Compute retrieval@K metrics for CLIP embedding retrieval.
    
    Given query embeddings (e.g., predicted from fMRI) and a gallery of ground truth
    embeddings (e.g., CLIP embeddings of all test images), compute how often the
    correct image appears in the top-K retrieved items.
    
    IMPORTANT: Both query and gallery MUST be L2-normalized to unit length.
    
    Scientific Context:
    - Standard evaluation for neural decoding (Ozcelik & VanRullen 2023)
    - Measures how well fMRI predictions capture semantic content
    - Higher R@K = better semantic alignment with CLIP space
    
    Args:
        query: Query embeddings, shape (n_queries, d)
               Typically: predictions from fMRI (after L2 normalization)
        gallery: Gallery embeddings, shape (n_gallery, d)
                 Typically: ground truth CLIP embeddings for all test stimuli
        gt_index: Ground truth indices, shape (n_queries,)
                  For each query i, gt_index[i] is the correct gallery index
        ks: Tuple of K values to compute retrieval@K for
    
    Returns:
        Dictionary with keys "R@1", "R@5", "R@10", etc.
        Values are retrieval rates in [0, 1]
    
    Example:
        >>> # 100 test samples, 1000 gallery images
        >>> query = model.predict(fmri_test)  # (100, 512), normalized
        >>> gallery = clip_cache.get_all()     # (1000, 512), normalized
        >>> gt_index = test_df["gallery_idx"].values  # (100,)
        >>> metrics = retrieval_at_k(query, gallery, gt_index, ks=(1, 5, 10))
        >>> print(f"R@1: {metrics['R@1']:.2%}, R@5: {metrics['R@5']:.2%}")
    
    Raises:
        ValueError: If shapes are incompatible or inputs not normalized
    """
    if query.shape[1] != gallery.shape[1]:
        raise ValueError(f"Dimension mismatch: query {query.shape[1]}D, gallery {gallery.shape[1]}D")
    
    if len(gt_index) != len(query):
        raise ValueError(f"Length mismatch: {len(query)} queries, {len(gt_index)} ground truth indices")
    
    # Verify normalization (warn if not normalized)
    query_norms = np.linalg.norm(query, axis=1)
    gallery_norms = np.linalg.norm(gallery, axis=1)
    
    if not np.allclose(query_norms, 1.0, atol=1e-3):
        logger.warning("Query vectors not L2-normalized (max deviation: {:.3f})".format(
            np.max(np.abs(query_norms - 1.0))
        ))
    if not np.allclose(gallery_norms, 1.0, atol=1e-3):
        logger.warning("Gallery vectors not L2-normalized (max deviation: {:.3f})".format(
            np.max(np.abs(gallery_norms - 1.0))
        ))
    
    # Compute similarity matrix
    sim = cosine_sim(query, gallery)  # (n_queries, n_gallery)
    
    # Argsort in descending order (most similar first)
    # ranks[i, :] contains gallery indices sorted by similarity to query i
    ranks = np.argsort(-sim, axis=1)  # (n_queries, n_gallery)
    
    # Compute retrieval@K for each K
    results = {}
    for k in ks:
        # For each query, check if ground truth appears in top-K
        top_k = ranks[:, :k]  # (n_queries, k)
        
        # Check if gt_index[i] is in top_k[i, :]
        hits = np.array([gt_index[i] in top_k[i] for i in range(len(query))])
        
        retrieval_rate = hits.mean()
        results[f"R@{k}"] = float(retrieval_rate)
    
    return results


def compute_ranking_metrics(
    query: np.ndarray,
    gallery: np.ndarray,
    gt_index: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive ranking metrics including mean rank and median rank.
    
    Args:
        query: Query embeddings, shape (n_queries, d), L2-normalized
        gallery: Gallery embeddings, shape (n_gallery, d), L2-normalized
        gt_index: Ground truth indices, shape (n_queries,)
    
    Returns:
        Dictionary with:
        - "mean_rank": Average rank of ground truth (lower is better)
        - "median_rank": Median rank of ground truth
        - "mrr": Mean reciprocal rank (higher is better, in [0, 1])
    """
    sim = cosine_sim(query, gallery)
    ranks = np.argsort(-sim, axis=1)
    
    # Find position of ground truth in ranked list
    gt_ranks = []
    for i in range(len(query)):
        gt_pos = np.where(ranks[i] == gt_index[i])[0][0]
        gt_ranks.append(gt_pos + 1)  # Convert to 1-based rank
    
    gt_ranks = np.array(gt_ranks)
    
    return {
        "mean_rank": float(gt_ranks.mean()),
        "median_rank": float(np.median(gt_ranks)),
        "mrr": float(np.mean(1.0 / gt_ranks)),  # Mean reciprocal rank
    }


def clip_score(generated_emb: np.ndarray, gt_emb: np.ndarray) -> np.ndarray:
    """
    Compute CLIPScore: per-sample cosine similarity between generated and GT embeddings.
    
    CLIPScore measures semantic similarity between generated images and their ground truths
    in CLIP embedding space. Higher scores indicate better semantic preservation.
    
    IMPORTANT: Both inputs MUST be L2-normalized to unit length.
    
    Scientific Context:
    - Standard metric for image generation quality (Hessel et al. 2021)
    - Measures semantic alignment without requiring pixel-level matching
    - Correlates well with human judgment of image quality
    
    Args:
        generated_emb: CLIP embeddings of generated images, shape (n_samples, d), L2-normalized
        gt_emb: CLIP embeddings of ground truth images, shape (n_samples, d), L2-normalized
    
    Returns:
        Per-sample cosine similarity, shape (n_samples,)
        Values in [-1, 1], typically [0, 1] for reasonable reconstructions
    
    Example:
        >>> # Evaluate 100 reconstructed images
        >>> gen_emb = encode_images(generated_images, clip_model)  # (100, 512), normalized
        >>> gt_emb = clip_cache.get(test_nsd_ids)  # (100, 512), normalized
        >>> scores = clip_score(gen_emb, gt_emb)  # (100,)
        >>> print(f"Mean CLIPScore: {scores.mean():.3f} ± {scores.std():.3f}")
    
    Raises:
        ValueError: If shapes are incompatible or inputs not normalized
    
    References:
        - Hessel et al. (2021). "CLIPScore: A Reference-free Evaluation Metric for Image Captioning"
    """
    if generated_emb.shape != gt_emb.shape:
        raise ValueError(f"Shape mismatch: generated {generated_emb.shape}, gt {gt_emb.shape}")
    
    # Verify normalization
    gen_norms = np.linalg.norm(generated_emb, axis=1)
    gt_norms = np.linalg.norm(gt_emb, axis=1)
    
    if not np.allclose(gen_norms, 1.0, atol=1e-3):
        logger.warning("Generated embeddings not L2-normalized (max deviation: {:.3f})".format(
            np.max(np.abs(gen_norms - 1.0))
        ))
    if not np.allclose(gt_norms, 1.0, atol=1e-3):
        logger.warning("GT embeddings not L2-normalized (max deviation: {:.3f})".format(
            np.max(np.abs(gt_norms - 1.0))
        ))
    
    # Per-sample dot product (cosine similarity when normalized)
    scores = np.sum(generated_emb * gt_emb, axis=1)
    
    return scores.astype(np.float32)
