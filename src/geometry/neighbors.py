"""Nearest neighbor utilities for geometry computations."""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Any, Optional


def build_knn_index(
    embeddings: np.ndarray,
    n_neighbors: int,
    metric: str = 'cosine'
) -> NearestNeighbors:
    """
    Build a k-nearest neighbors index.
    
    Args:
        embeddings: 2D array of embeddings (n_samples, n_features)
        n_neighbors: Number of neighbors to find
        metric: Distance metric ('cosine', 'euclidean', etc.)
        
    Returns:
        Fitted NearestNeighbors object
    """
    # Add 1 to n_neighbors since the point itself will be the first neighbor
    knn = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        metric=metric,
        algorithm='auto'
    )
    knn.fit(embeddings)
    return knn


def get_neighbors(
    index: NearestNeighbors,
    point_index: int,
    k: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get k nearest neighbors for a point.
    
    Args:
        index: Fitted NearestNeighbors object
        point_index: Index of the query point
        k: Number of neighbors (excluding the point itself)
           If None, uses the index's configured n_neighbors
        
    Returns:
        Tuple of (neighbor_indices, neighbor_distances)
        Both exclude the query point itself
    """
    # Get the point embedding
    point = index._fit_X[point_index:point_index+1]
    
    # Query for neighbors
    if k is None:
        distances, indices = index.kneighbors(point)
    else:
        distances, indices = index.kneighbors(point, n_neighbors=k+1)
    
    # Remove the point itself (first neighbor with distance 0)
    indices = indices[0, 1:]
    distances = distances[0, 1:]
    
    return indices, distances


def get_neighbors_batch(
    index: NearestNeighbors,
    point_indices: np.ndarray,
    k: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get k nearest neighbors for multiple points.
    
    Args:
        index: Fitted NearestNeighbors object
        point_indices: Array of point indices to query
        k: Number of neighbors (excluding the points themselves)
        
    Returns:
        Tuple of (neighbor_indices, neighbor_distances)
        Shape: (n_points, k)
    """
    points = index._fit_X[point_indices]
    
    if k is None:
        distances, indices = index.kneighbors(points)
    else:
        distances, indices = index.kneighbors(points, n_neighbors=k+1)
    
    # Remove the points themselves
    indices = indices[:, 1:]
    distances = distances[:, 1:]
    
    return indices, distances
