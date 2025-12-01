"""Reference corpus builder for improved geometry features."""

import numpy as np
from pathlib import Path
from typing import Optional
import json

from sklearn.decomposition import PCA


def build_reference_corpus(
    texts: list[str],
    embedding_client,
    output_dir: Path,
    n_samples: int = 10000,
    pca_components: int = 50,
    seed: int = 42
) -> dict:
    """
    Build a reference corpus for geometry normalization.
    
    Args:
        texts: List of reference texts
        embedding_client: Client for generating embeddings
        output_dir: Directory to save reference data
        n_samples: Number of samples to use
        pca_components: Number of PCA components to fit
        seed: Random seed
        
    Returns:
        Dictionary with reference statistics
    """
    np.random.seed(seed)
    
    # Sample if needed
    if len(texts) > n_samples:
        indices = np.random.choice(len(texts), n_samples, replace=False)
        texts = [texts[i] for i in indices]
    
    print(f"Building reference corpus with {len(texts)} samples...")
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = embedding_client.embed_texts(texts)
    
    # Compute statistics
    print("Computing statistics...")
    mean_embedding = np.mean(embeddings, axis=0)
    std_embedding = np.std(embeddings, axis=0)
    
    # Fit PCA
    print(f"Fitting PCA with {pca_components} components...")
    pca = PCA(n_components=pca_components)
    pca.fit(embeddings)
    
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"PCA explained variance: {explained_var:.3f}")
    
    # Save everything
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "reference_embeddings.npy", embeddings)
    np.save(output_dir / "mean_embedding.npy", mean_embedding)
    np.save(output_dir / "std_embedding.npy", std_embedding)
    np.save(output_dir / "pca_components.npy", pca.components_)
    np.save(output_dir / "pca_mean.npy", pca.mean_)
    np.save(output_dir / "pca_explained_variance.npy", pca.explained_variance_)
    
    # Save metadata
    metadata = {
        "n_samples": len(texts),
        "embedding_dim": embeddings.shape[1],
        "pca_components": pca_components,
        "pca_explained_variance": float(explained_var),
        "seed": seed
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Reference corpus saved to {output_dir}")
    
    return {
        "embeddings": embeddings,
        "mean": mean_embedding,
        "std": std_embedding,
        "pca": pca,
        "metadata": metadata
    }


def load_reference_corpus(reference_dir: Path) -> dict:
    """Load a previously built reference corpus."""
    embeddings = np.load(reference_dir / "reference_embeddings.npy")
    mean_embedding = np.load(reference_dir / "mean_embedding.npy")
    std_embedding = np.load(reference_dir / "std_embedding.npy")
    
    # Reconstruct PCA
    components = np.load(reference_dir / "pca_components.npy")
    pca_mean = np.load(reference_dir / "pca_mean.npy")
    explained_variance = np.load(reference_dir / "pca_explained_variance.npy")
    
    with open(reference_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    pca = PCA(n_components=metadata["pca_components"])
    pca.components_ = components
    pca.mean_ = pca_mean
    pca.explained_variance_ = explained_variance
    pca.explained_variance_ratio_ = explained_variance / np.sum(explained_variance)
    
    return {
        "embeddings": embeddings,
        "mean": mean_embedding,
        "std": std_embedding,
        "pca": pca,
        "metadata": metadata
    }


if __name__ == "__main__":
    """Build reference corpus from benchmark questions."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.models.embedding_client import EmbeddingClient
    from src.utils.io import read_jsonl
    from src.config import load_config
    
    # Load config
    config = load_config("experiments/config_example.yaml")
    
    # Collect all benchmark texts
    prompts_dir = Path("data/prompts")
    all_texts = []
    
    for jsonl_file in prompts_dir.glob("*.jsonl"):
        data = read_jsonl(jsonl_file)
        all_texts.extend([item["question"] for item in data])
    
    print(f"Collected {len(all_texts)} texts from benchmark")
    
    # Initialize embedding client
    embedding_client = EmbeddingClient(
        model_name=config.embedding_model,
        batch_size=config.embedding_batch_size
    )
    
    # Build corpus
    reference_dir = Path("data/reference_corpus")
    build_reference_corpus(
        texts=all_texts,
        embedding_client=embedding_client,
        output_dir=reference_dir,
        n_samples=min(len(all_texts), 10000),
        pca_components=50,
        seed=config.seed
    )
