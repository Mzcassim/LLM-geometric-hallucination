"""Risk manifold visualizations using UMAP and t-SNE.

Creates 2D projections of the embedding space with hallucination risk overlays.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.seed import set_seed


def load_embeddings_and_data():
    """Load question embeddings and results data."""
    
    embeddings_file = Path("data/processed/question_embeddings.npy")
    results_file = Path("results/all_results.csv")
    
    if not embeddings_file.exists():
        print(f"Error: {embeddings_file} not found")
        return None, None
    
    embeddings = np.load(embeddings_file)
    df = pd.read_csv(results_file)
    
    print(f"Loaded {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    print(f"Loaded {len(df)} results")
    
    return embeddings, df


def create_risk_manifold(embeddings, df, output_dir, method='umap', seed=42):
    """Create 2D projection with risk overlays."""
    
    print(f"\nCreating {method.upper()} projection...")
    
    # Project to 2D
    if method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
    else:  # t-SNE
        reducer = TSNE(n_components=2, random_state=seed, perplexity=30)
    
    coords_2d = reducer.fit_transform(embeddings)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Color by category
    ax = axes[0]
    categories = df['category'].values
    category_colors = {'factual': 'green', 'impossible': 'orange', 
                       'nonexistent': 'red', 'ambiguous': 'blue'}
    
    for category, color in category_colors.items():
        mask = categories == category
        if mask.sum() > 0:
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], 
                      c=color, label=category.title(), alpha=0.6, s=20)
    
    ax.set_title(f'{method.upper()} Projection - By Category', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')
    ax.grid(alpha=0.3)
    
    # 2. Color by hallucination status
    ax = axes[1]
    hallucinated = df['is_hallucinated'].values
    
    # Plot correct first (background)
    correct_mask = hallucinated == 0
    ax.scatter(coords_2d[correct_mask, 0], coords_2d[correct_mask, 1], 
              c='lightgray', label='Correct', alpha=0.4, s=15)
    
    # Plot hallucinated on top (foreground)
    hall_mask = hallucinated == 1
    ax.scatter(coords_2d[hall_mask, 0], coords_2d[hall_mask, 1], 
              c='red', label='Hallucinated', alpha=0.8, s=40, marker='X')
    
    ax.set_title(f'{method.upper()} Projection - Hallucination Risk', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')
    ax.grid(alpha=0.3)
    
    # 3. Color by density
    ax = axes[2]
    if 'density' in df.columns:
        density = df['density'].values
        scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                            c=density, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax, label='Density (Higher = More Crowded)')
        ax.set_title(f'{method.upper()} Projection - Density', fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Density not available', transform=ax.transAxes,
                ha='center', va='center')
        ax.set_title(f'{method.upper()} Projection', fontsize=14, fontweight='bold')
    
    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / f"risk_manifold_{method}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file}")
    
    return coords_2d


def create_category_manifolds(embeddings, df, coords_2d, output_dir, method='umap'):
    """Create separate manifold for each category."""
    
    print(f"\nCreating per-category manifolds...")
    
    categories = df['category'].unique()
    n_cats = len(categories)
    
    fig, axes = plt.subplots(1, n_cats, figsize=(5*n_cats, 4))
    
    if n_cats == 1:
        axes = [axes]
    
    for ax, category in zip(axes, categories):
        mask = df['category'] == category
        coords_cat = coords_2d[mask]
        hallucinated_cat = df[mask]['is_hallucinated'].values
        
        # Plot correct first
        correct_mask = hallucinated_cat == 0
        ax.scatter(coords_cat[correct_mask, 0], coords_cat[correct_mask, 1], 
                  c='lightblue', label='Correct', alpha=0.5, s=30)
        
        # Plot hallucinated on top
        hall_mask = hallucinated_cat == 1
        if hall_mask.sum() > 0:
            ax.scatter(coords_cat[hall_mask, 0], coords_cat[hall_mask, 1], 
                      c='red', label='Hallucinated', alpha=0.8, s=50, marker='X')
        
        hall_rate = hallucinated_cat.mean()
        ax.set_title(f'{category.title()}\n({hall_rate:.0%} hallucinated)', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.set_xlabel(f'{method.upper()} 1', fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel(f'{method.upper()} 2', fontsize=10)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / f"category_manifolds_{method}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file}")


def create_geometry_heatmaps(coords_2d, df, output_dir, method='umap'):
    """Create heatmaps of geometric features on the manifold."""
    
    print(f"\nCreating geometry heatmaps...")
    
    geometry_features = ['local_id', 'curvature_score', 'oppositeness_score', 
                         'density', 'centrality']
    available = [f for f in geometry_features if f in df.columns]
    
    if len(available) == 0:
        print("No geometry features available for heatmaps")
        return
    
    fig, axes = plt.subplots(1, len(available), figsize=(5*len(available), 4))
    
    if len(available) == 1:
        axes = [axes]
    
    for ax, feat in zip(axes, available):
        values = df[feat].values
        
        # Create heatmap
        scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                            c=values, cmap='coolwarm', alpha=0.7, s=25)
        
        plt.colorbar(scatter, ax=ax, label=feat.replace('_', ' ').title())
        ax.set_title(feat.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{method.upper()} 1')
        if ax == axes[0]:
            ax.set_ylabel(f'{method.upper()} 2')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / f"geometry_heatmaps_{method}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file}")


def main():
    """Generate all risk manifold visualizations."""
    
    set_seed(42)
    
    # Load data
    embeddings, df = load_embeddings_and_data()
    if embeddings is None:
        return
    
    # Create output directory
    output_dir = Path("results/v3/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("GENERATING RISK MANIFOLD VISUALIZATIONS")
    print("="*60)
    
    # UMAP projections
    print("\n--- UMAP ---")
    coords_umap = create_risk_manifold(embeddings, df, output_dir, method='umap', seed=42)
    create_category_manifolds(embeddings, df, coords_umap, output_dir, method='umap')
    create_geometry_heatmaps(coords_umap, df, output_dir, method='umap')
    
    # t-SNE projections
    print("\n--- t-SNE ---")
    coords_tsne = create_risk_manifold(embeddings, df, output_dir, method='tsne', seed=42)
    create_category_manifolds(embeddings, df, coords_tsne, output_dir, method='tsne')
    create_geometry_heatmaps(coords_tsne, df, output_dir, method='tsne')
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print(f"All figures saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
