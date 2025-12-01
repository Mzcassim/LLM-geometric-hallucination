"""Centrality measures for embeddings."""

import numpy as np


def compute_distance_to_center(
    embeddings: np.ndarray,
    center: np.ndarray,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Compute distance from each embedding to the global center.
    
    Higher values = more peripheral/extreme positions
    Lower values = more central/typical
    
    Args:
        embeddings: Query embeddings (n, dim)
        center: Center point (dim,) - typically mean of reference corpus
        metric: Distance metric ('euclidean' or 'cosine')
        
    Returns:
        Distances to center (n,)
    """
    if metric == 'euclidean':
        # Euclidean distance
        distances = np.linalg.norm(embeddings - center, axis=1)
    elif metric == 'cosine':
        # Cosine distance = 1 - cosine_similarity
        # Normalize vectors
        emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        center_norm = center / (np.linalg.norm(center) + 1e-10)
        
        # Cosine similarity
        cos_sim = np.dot(emb_norm, center_norm)
        
        # Cosine distance
        distances = 1 - cos_sim
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return distances


def normalize_centrality(
    distances: np.ndarray,
    reference_mean: float,
    reference_std: float
) -> np.ndarray:
    """
    Z-score normalization of centrality.
    
    Args:
        distances: Raw distance values
        reference_mean: Mean distance of reference corpus
        reference_std: Std distance of reference corpus
        
    Returns:
        Normalized distances (z-scores)
    """
    return (distances - reference_mean) / (reference_std + 1e-10)


def compute_radial_percentile(
    distances: np.ndarray,
    reference_distances: np.ndarray
) -> np.ndarray:
    """
    Convert distances to percentiles.
    
    Higher percentile = more peripheral
    
    Args:
        distances: Query distances
        reference_distances: Reference corpus distances
        
    Returns:
        Percentiles (0-100)
    """
    percentiles = np.zeros(len(distances))
    
    for i, d in enumerate(distances):
        percentile = (reference_distances < d).sum() / len(reference_distances) * 100
        percentiles[i] = percentile
    
    return percentiles
