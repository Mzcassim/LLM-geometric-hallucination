"""Intrinsic dimension estimation using TwoNN method."""

import numpy as np
from typing import Tuple
from .neighbors import build_knn_index, get_neighbors_batch


def estimate_twonn_id(distances: np.ndarray) -> float:
    """
    Estimate intrinsic dimension using TwoNN method.
    
    The TwoNN (Two Nearest Neighbors) estimator uses the ratio of distances
    to the first and second nearest neighbors.
    
    Args:
        distances: Array of distances to neighbors for a single point,
                   sorted in ascending order. Shape: (n_neighbors,)
        
    Returns:
        Estimated local intrinsic dimension
    """
    if len(distances) < 2:
        return np.nan
    
    r1 = distances[0]  # Distance to 1st nearest neighbor
    r2 = distances[1]  # Distance to 2nd nearest neighbor
    
    # Avoid division by zero or log of zero
    if r1 == 0 or r2 == 0 or r1 >= r2:
        return np.nan
    
    # TwoNN estimator: d ≈ -1 / log(r1/r2)
    # More stable form: d ≈ 1 / log(r2/r1)
    ratio = r2 / r1
    
    if ratio <= 1:
        return np.nan
    
    dimension = 1.0 / np.log(ratio)
    
    return dimension


def estimate_twonn_id_mle(distances_matrix: np.ndarray) -> float:
    """
    Estimate intrinsic dimension using TwoNN MLE over multiple points.
    
    This uses the maximum likelihood estimator combining information
    from multiple points.
    
    Args:
        distances_matrix: 2D array of distances, shape (n_points, n_neighbors)
                         Each row contains sorted distances for one point
        
    Returns:
        Estimated intrinsic dimension (scalar)
    """
    n_points = distances_matrix.shape[0]
    
    log_ratios = []
    for i in range(n_points):
        distances = distances_matrix[i]
        if len(distances) < 2:
            continue
        
        r1 = distances[0]
        r2 = distances[1]
        
        if r1 > 0 and r2 > r1:
            log_ratios.append(np.log(r2 / r1))
    
    if len(log_ratios) == 0:
        return np.nan
    
    # MLE: d = 1 / mean(log(r2/r1))
    mean_log_ratio = np.mean(log_ratios)
    
    if mean_log_ratio <= 0:
        return np.nan
    
    dimension = 1.0 / mean_log_ratio
    
    return dimension


def compute_local_id_for_all(
    embeddings: np.ndarray,
    n_neighbors: int = 20,
    metric: str = 'cosine'
) -> np.ndarray:
    """
    Compute local intrinsic dimension for all points.
    
    Args:
        embeddings: 2D array of embeddings (n_samples, n_features)
        n_neighbors: Number of neighbors to use for estimation
        metric: Distance metric to use
        
    Returns:
        1D array of local intrinsic dimensions (n_samples,)
    """
    n_points = embeddings.shape[0]
    
    # Build k-NN index
    knn_index = build_knn_index(embeddings, n_neighbors, metric=metric)
    
    # Get neighbors for all points
    point_indices = np.arange(n_points)
    _, distances = get_neighbors_batch(knn_index, point_indices, k=n_neighbors)
    
    # Estimate ID for each point
    local_ids = np.zeros(n_points)
    for i in range(n_points):
        local_ids[i] = estimate_twonn_id(distances[i])
    
    return local_ids
