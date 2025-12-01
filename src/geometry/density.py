"""Local density estimation for embeddings."""

import numpy as np
from typing import Optional
from sklearn.neighbors import NearestNeighbors


def compute_local_density(
    embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    k: int = 20,
    metric: str = 'cosine'
) -> np.ndarray:
    """
    Compute local density as inverse of mean distance to k nearest neighbors.
    
    Lower density = more isolated/out-of-distribution
    Higher density = more central/supported by training data
    
    Args:
        embeddings: Query embeddings (n_queries, dim)
        reference_embeddings: Reference corpus (n_ref, dim)
        k: Number of nearest neighbors
        metric: Distance metric
        
    Returns:
        Density scores (n_queries,) - higher = denser region
    """
    # Build k-NN index on reference
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric, algorithm='auto')
    nbrs.fit(reference_embeddings)
    
    # Find k nearest neighbors for each query
    distances, _ = nbrs.kneighbors(embeddings)
    
    # Density = 1 / mean_distance
    # Add small epsilon to avoid division by zero
    mean_distances = np.mean(distances, axis=1)
    density = 1.0 / (mean_distances + 1e-10)
    
    return density


def compute_density_percentile(
    density: np.ndarray,
    reference_density: np.ndarray
) -> np.ndarray:
    """
    Convert density to percentile relative to reference distribution.
    
    Lower percentile = more out-of-distribution
    
    Args:
        density: Query densities
        reference_density: Reference corpus densities
        
    Returns:
        Percentiles (0-100)
    """
    percentiles = np.zeros(len(density))
    
    for i, d in enumerate(density):
        percentile = (reference_density < d).sum() / len(reference_density) * 100
        percentiles[i] = percentile
    
    return percentiles


def normalize_density(
    density: np.ndarray,
    reference_mean: float,
    reference_std: float
) -> np.ndarray:
    """
    Z-score normalization of density.
    
    Args:
        density: Raw density values
        reference_mean: Mean density of reference corpus
        reference_std: Std density of reference corpus
        
    Returns:
        Normalized density (z-scores)
    """
    return (density - reference_mean) / (reference_std + 1e-10)
