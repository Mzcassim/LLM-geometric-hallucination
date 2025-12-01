"""Curvature proxy via local PCA distortion."""

import numpy as np
from sklearn.decomposition import PCA
from typing import Optional
from .neighbors import build_knn_index, get_neighbors


def compute_pca_residual_variance(
    points: np.ndarray,
    n_components: Optional[int] = None
) -> float:
    """
    Compute residual variance after PCA.
    
    Higher residual = more curvature/irregularity.
    
    Args:
        points: 2D array of points (n_points, n_features)
        n_components: Number of PCA components to keep
                     If None, keeps min(n_points, n_features) // 2
        
    Returns:
        Residual variance (1 - explained_variance_ratio)
    """
    if len(points) < 2:
        return np.nan
    
    # Center the points
    centered = points - points.mean(axis=0)
    
    # Determine number of components
    if n_components is None:
        n_components = min(points.shape[0], points.shape[1]) // 2
    
    n_components = min(n_components, points.shape[0] - 1, points.shape[1])
    
    if n_components < 1:
        return np.nan
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    try:
        pca.fit(centered)
        
        # Calculate explained variance
        explained_var = np.sum(pca.explained_variance_ratio_)
        residual_var = 1.0 - explained_var
        
        return residual_var
    
    except Exception as e:
        # In case of numerical issues
        return np.nan


def compute_curvature_proxy(
    embeddings: np.ndarray,
    local_ids: np.ndarray,
    n_neighbors: int = 30,
    metric: str = 'cosine'
) -> np.ndarray:
    """
    Compute curvature proxy for all points using local PCA distortion.
    
    Args:
        embeddings: 2D array of embeddings (n_samples, n_features)
        local_ids: Array of local intrinsic dimensions (n_samples,)
        n_neighbors: Number of neighbors to use for local neighborhood
        metric: Distance metric
        
    Returns:
        1D array of curvature scores (n_samples,)
        Higher values indicate more curvature/irregularity
    """
    n_points = embeddings.shape[0]
    curvature_scores = np.zeros(n_points)
    
    # Build k-NN index
    knn_index = build_knn_index(embeddings, n_neighbors, metric=metric)
    
    for i in range(n_points):
        # Get neighbors
        neighbor_indices, _ = get_neighbors(knn_index, i, k=n_neighbors)
        
        # Get neighborhood points (including the point itself)
        neighborhood = embeddings[np.concatenate([[i], neighbor_indices])]
        
        # Determine number of PCA components based on local ID
        local_id = local_ids[i]
        if np.isnan(local_id):
            n_components = n_neighbors // 2
        else:
            # Use the estimated local dimension, capped appropriately
            n_components = int(np.clip(local_id, 1, n_neighbors))
        
        # Compute residual variance
        residual = compute_pca_residual_variance(neighborhood, n_components)
        curvature_scores[i] = residual
    
    return curvature_scores
