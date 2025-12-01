"""Compute geometric features for embeddings."""

import sys
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import load_config, ProjectConfig
from src.models.embedding_client import EmbeddingClient
from src.geometry.intrinsic_dimension import compute_local_id_for_all
from src.geometry.curvature import compute_curvature_proxy
from src.geometry.oppositeness import fit_global_pca, compute_oppositeness_scores
from src.geometry.reference_corpus import load_reference_corpus
from src.geometry.density import compute_local_density
from src.geometry.centrality import compute_distance_to_center
from src.utils.io import read_jsonl
from src.utils.logging_utils import setup_logger
from src.utils.seed import set_seed


def compute_geometry(config: ProjectConfig):
    """Compute geometric features for all questions."""
    logger = setup_logger(
        "geometry",
        log_file=config.data_dir / "logs" / "geometry.log"
    )
    
    # Set random seed
    set_seed(config.seed)
    
    logger.info(f"Starting geometry computation with embedding model: {config.embedding_model}")
    
    # Initialize embedding client
    embed_client = EmbeddingClient(
        model_name=config.embedding_model,
        batch_size=config.batch_size,
        max_retries=config.max_retries,
        timeout=config.api_timeout
    )
    
    # Load model answers (contains questions)
    answers_file = config.processed_dir / "model_answers.jsonl"
    if not answers_file.exists():
        logger.error(f"Model answers file not found: {answers_file}")
        logger.error("Please run run_generation.py first")
        return
    
    answers = read_jsonl(answers_file)
    logger.info(f"Loaded {len(answers)} questions")
    
    # Extract questions and IDs
    question_ids = [a["id"] for a in answers]
    questions = [a["question"] for a in answers]
    categories = [a["category"] for a in answers]
    
    # Compute embeddings for questions
    logger.info("Computing question embeddings...")
    question_embeddings = embed_client.embed_texts(questions)
    logger.info(f"Question embeddings shape: {question_embeddings.shape}")
    
    # Save embeddings
    embeddings_file = config.processed_dir / "question_embeddings.npy"
    np.save(embeddings_file, question_embeddings)
    logger.info(f"Saved embeddings to {embeddings_file}")
    
    # Save ID mapping
    id_mapping = {qid: i for i, qid in enumerate(question_ids)}
    mapping_file = config.processed_dir / "embedding_id_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(id_mapping, f, indent=2)
    logger.info(f"Saved ID mapping to {mapping_file}")
    
    # Load reference corpus for V2 features
    reference_dir = config.data_dir / "reference_corpus"
    if reference_dir.exists():
        logger.info("Loading reference corpus...")
        ref_corpus = load_reference_corpus(reference_dir)
        ref_embeddings = ref_corpus["embeddings"]
        ref_mean = ref_corpus["mean"]
    else:
        logger.warning("Reference corpus not found! Using self as reference (not recommended for production).")
        ref_embeddings = question_embeddings
        ref_mean = np.mean(question_embeddings, axis=0)
    
    # Compute local intrinsic dimension
    logger.info("Computing local intrinsic dimensions...")
    local_ids = compute_local_id_for_all(
        question_embeddings,
        n_neighbors=config.n_neighbors_id,
        metric='cosine'
    )
    logger.info(f"Local ID range: {np.nanmin(local_ids):.2f} to {np.nanmax(local_ids):.2f}")
    
    # Compute curvature proxy
    logger.info("Computing curvature scores...")
    curvature_scores = compute_curvature_proxy(
        question_embeddings,
        local_ids,
        n_neighbors=config.n_neighbors_curvature,
        metric='cosine'
    )
    logger.info(f"Curvature range: {np.nanmin(curvature_scores):.4f} to {np.nanmax(curvature_scores):.4f}")
    
    # Compute oppositeness scores
    logger.info("Computing oppositeness scores...")
    global_pca = fit_global_pca(question_embeddings, n_components=config.n_pca_components)
    logger.info(f"Global PCA explained variance: {global_pca.explained_variance_ratio_.sum():.3f}")
    
    oppositeness_scores = compute_oppositeness_scores(
        question_embeddings,
        global_pca,
        n_flip=config.n_flip_components
    )
    logger.info(f"Oppositeness range: {np.nanmin(oppositeness_scores):.4f} to {np.nanmax(oppositeness_scores):.4f}")
    
    # Compute Density (V2)
    logger.info("Computing local density...")
    density_scores = compute_local_density(
        question_embeddings,
        ref_embeddings,
        k=config.n_neighbors_id,
        metric='cosine'
    )
    logger.info(f"Density range: {np.nanmin(density_scores):.4f} to {np.nanmax(density_scores):.4f}")
    
    # Compute Centrality (V2)
    logger.info("Computing centrality...")
    centrality_scores = compute_distance_to_center(
        question_embeddings,
        ref_mean,
        metric='cosine'
    )
    logger.info(f"Centrality range: {np.nanmin(centrality_scores):.4f} to {np.nanmax(centrality_scores):.4f}")
    
    # Create features dataframe
    features_df = pd.DataFrame({
        'id': question_ids,
        'category': categories,
        'local_id': local_ids,
        'curvature_score': curvature_scores,
        'oppositeness_score': oppositeness_scores,
        'density': density_scores,
        'centrality': centrality_scores
    })
    
    # Save features
    features_file = config.processed_dir / "geometry_features.csv"
    features_df.to_csv(features_file, index=False)
    logger.info(f"Saved geometry features to {features_file}")
    
    # Log summary by category
    logger.info("\nGeometry features by category:")
    for category in features_df['category'].unique():
        cat_data = features_df[features_df['category'] == category]
        logger.info(f"\n{category}:")
        logger.info(f"  Local ID: mean={cat_data['local_id'].mean():.2f}, std={cat_data['local_id'].std():.2f}")
        logger.info(f"  Curvature: mean={cat_data['curvature_score'].mean():.4f}, std={cat_data['curvature_score'].std():.4f}")
        logger.info(f"  Oppositeness: mean={cat_data['oppositeness_score'].mean():.4f}, std={cat_data['oppositeness_score'].std():.4f}")
        logger.info(f"  Density: mean={cat_data['density'].mean():.4f}, std={cat_data['density'].std():.4f}")
        logger.info(f"  Centrality: mean={cat_data['centrality'].mean():.4f}, std={cat_data['centrality'].std():.4f}")
    
    logger.info("\nGeometry computation complete!")


def main():
    parser = argparse.ArgumentParser(description="Compute geometry features")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/config_example.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    compute_geometry(config)


if __name__ == "__main__":
    main()
