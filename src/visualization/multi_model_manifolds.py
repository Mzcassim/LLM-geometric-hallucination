"""Generate V3 visualizations for multiple models.

Creates risk manifolds and geometry heatmaps for each model in the multi-model experiment.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.seed import set_seed


def load_data(results_file):
    """Load aggregated multi-model results and embeddings."""
    df = pd.read_csv(results_file)
    
    # Load embeddings (same for all models as prompts are same)
    # We need to match embeddings to the rows. 
    # Since df has multiple rows per prompt (one per model), we need to be careful.
    
    # Load original embeddings
    embeddings_file = Path("data/processed/question_embeddings.npy")
    if not embeddings_file.exists():
        print("Embeddings file not found")
        return None, None
        
    embeddings = np.load(embeddings_file)
    
    # We assume the embeddings correspond to the prompts in V2 results.
    # We need a mapping from ID to embedding index.
    # Let's load V2 results to get the ID order
    v2_results = pd.read_csv("archive/v2_production_run/results/all_results.csv")
    id_to_idx = {row['id']: i for i, row in v2_results.iterrows()}
    
    return df, embeddings, id_to_idx


def create_multi_model_manifolds(df, embeddings, id_to_idx, output_dir):
    """Create risk manifolds for each model."""
    
    print("Computing UMAP projection (based on unique prompts)...")
    
    # Compute UMAP on unique embeddings once
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords_2d = reducer.fit_transform(embeddings)
    
    models = df['model_name'].unique()
    print(f"Generating plots for {len(models)} models...")
    
    # Create a grid of plots
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten()
    
    for i, model in enumerate(models):
        ax = axes[i]
        model_df = df[df['model_name'] == model]
        
        # Get coordinates for these specific prompts
        indices = [id_to_idx[pid] for pid in model_df['id'] if pid in id_to_idx]
        if not indices:
            continue
            
        model_coords = coords_2d[indices]
        hallucinated = model_df[model_df['id'].isin(id_to_idx)]['is_hallucinated'].values
        
        # Plot correct
        correct_mask = hallucinated == 0
        ax.scatter(model_coords[correct_mask, 0], model_coords[correct_mask, 1],
                  c='lightgray', alpha=0.3, s=10, label='Correct')
        
        # Plot hallucinated
        hall_mask = hallucinated == 1
        ax.scatter(model_coords[hall_mask, 0], model_coords[hall_mask, 1],
                  c='red', alpha=0.7, s=25, marker='x', label='Hallucinated')
        
        hall_rate = hallucinated.mean()
        ax.set_title(f"{model}\n({hall_rate:.1%} Hallucination)", fontsize=10, fontweight='bold')
        
        if i == 0:
            ax.legend(loc='best', fontsize=8)
            
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    output_file = Path(output_dir) / "multi_model_risk_manifolds.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate multi-model visualizations")
    parser.add_argument("--results-file", type=str, default="results/v3/multi_model/all_models_results.csv")
    parser.add_argument("--output-dir", type=str, default="results/v3/multi_model/figures")
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    df, embeddings, id_to_idx = load_data(args.results_file)
    if df is not None:
        create_multi_model_manifolds(df, embeddings, id_to_idx, args.output_dir)


if __name__ == "__main__":
    main()
