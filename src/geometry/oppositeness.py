"""Geometric oppositeness metric using PCA-based direction flipping."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def fit_global_pca(
    embeddings: np.ndarray,
    n_components: int = 10
) -> PCA:
    """
    Fit global PCA to embeddings.
    
    Args:
        embeddings: 2D array of embeddings (n_samples, n_features)
        n_components: Number of principal components to keep
        
    Returns:
        Fitted PCA object
    """
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    return pca


def compute_oppositeness_scores(
    embeddings: np.ndarray,
    pca: PCA,
    n_flip: int = 3
) -> np.ndarray:
    """
    Compute geometric oppositeness scores.
    
    For each point, we:
    1. Project into PCA space
    2. Flip the top n_flip components
    3. Project back to original space
    4. Measure distance to nearest real point
    
    Higher distance = point is in a region where opposites are far from data
    This might indicate a more "extreme" or boundary region.
    
    Args:
        embeddings: 2D array of embeddings (n_samples, n_features)
        pca: Fitted PCA object
        n_flip: Number of top components to flip
        
    Returns:
        1D array of oppositeness scores (n_samples,)
    """
    n_points = embeddings.shape[0]
    n_flip = min(n_flip, pca.n_components_)
    
    # Transform to PCA space
    pca_coords = pca.transform(embeddings)
    
    # Create opposite points by flipping signs of top components
    opposite_coords = pca_coords.copy()
    opposite_coords[:, :n_flip] *= -1
    
    # Transform back to original space
    opposite_points = pca.inverse_transform(opposite_coords)
    
    # Build k-NN index for finding nearest real neighbors
    knn = NearestNeighbors(n_neighbors=1, metric='cosine')
    knn.fit(embeddings)
    
    # Find distance from each opposite point to nearest real point
    distances, _ = knn.kneighbors(opposite_points)
    oppositeness_scores = distances.flatten()
    
    return oppositeness_scores
