"""Analysis and visualization notebook/script."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import load_config
from src.evaluation.metrics import (
    compute_correlations,
    descriptive_stats_by_category,
    compute_hallucination_rates_by_bins
)


def create_visualizations(config_path: str = "experiments/config_example.yaml"):
    """Create visualizations for the analysis."""
    config = load_config(config_path)
    
    # Load results
    results_file = config.results_dir / "all_results.csv"
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Please run the full pipeline first")
        return
    
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} results")
    
    # Set up plotting style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    
    # Create figures directory
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Define geometry features
    geo_features = ['local_id', 'curvature_score', 'oppositeness_score']
    
    # 1. Distribution plots by hallucination status
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, feature in enumerate(geo_features):
        ax = axes[i]
        
        # Plot distributions
        hallucinated = df[df['is_hallucinated'] == 1][feature].dropna()
        not_hallucinated = df[df['is_hallucinated'] == 0][feature].dropna()
        
        ax.hist(not_hallucinated, alpha=0.5, label='Not Hallucinated', bins=20, density=True)
        ax.hist(hallucinated, alpha=0.5, label='Hallucinated', bins=20, density=True)
        ax.set_xlabel(feature.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_title(f'{feature.replace("_", " ").title()} Distribution')
    
    plt.tight_layout()
    plt.savefig(config.figures_dir / "geometry_distributions.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {config.figures_dir / 'geometry_distributions.png'}")
    plt.close()
    
    # 2. Scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, feature in enumerate(geo_features):
        ax = axes[i]
        
        # Scatter plot with jitter
        valid_df = df[[feature, 'is_hallucinated']].dropna()
        x = valid_df[feature]
        y = valid_df['is_hallucinated'] + np.random.normal(0, 0.02, size=len(valid_df))
        
        ax.scatter(x, y, alpha=0.3, s=20)
        ax.set_xlabel(feature.replace('_', ' ').title())
        ax.set_ylabel('Hallucinated (with jitter)')
        ax.set_ylim(-0.2, 1.2)
        
        # Add correlation to title
        if len(valid_df) > 1:
            corr, p = spearmanr(valid_df[feature], valid_df['is_hallucinated'])
            ax.set_title(f'{feature.replace("_", " ").title()}\nSpearman r={corr:.3f}, p={p:.3f}')
    
    plt.tight_layout()
    plt.savefig(config.figures_dir / "geometry_scatter.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {config.figures_dir / 'geometry_scatter.png'}")
    plt.close()
    
    # 3. Box plots by category
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, feature in enumerate(geo_features):
        ax = axes[i]
        
        # Box plot
        categories = df['category'].unique()
        data_to_plot = [df[df['category'] == cat][feature].dropna() for cat in categories]
        
        ax.boxplot(data_to_plot, labels=categories)
        ax.set_ylabel(feature.replace('_', ' ').title())
        ax.set_xlabel('Category')
        ax.set_title(f'{feature.replace("_", " ").title()} by Category')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(config.figures_dir / "geometry_by_category.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {config.figures_dir / 'geometry_by_category.png'}")
    plt.close()
    
    # 4. Hallucination rates by geometry bins
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, feature in enumerate(geo_features):
        ax = axes[i]
        
        bins_df = compute_hallucination_rates_by_bins(df, feature, n_bins=5)
        
        if len(bins_df) > 0:
            ax.bar(range(len(bins_df)), bins_df['hallucination_rate'])
            ax.set_xlabel('Bin (low to high)')
            ax.set_ylabel('Hallucination Rate')
            ax.set_title(f'Hallucination Rate by {feature.replace("_", " ").title()} Bins')
            ax.set_ylim(0, 1)
            
            # Add bin ranges as labels
            labels = [f'{row["bin_min"]:.2f}-{row["bin_max"]:.2f}' for _, row in bins_df.iterrows()]
            ax.set_xticks(range(len(bins_df)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(config.figures_dir / "hallucination_by_bins.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {config.figures_dir / 'hallucination_by_bins.png'}")
    plt.close()
    
    # Create tables
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Correlation table
    corr_df = compute_correlations(df, geo_features, target='is_hallucinated')
    corr_df.to_csv(config.tables_dir / "correlations.csv", index=False)
    print(f"\nSaved: {config.tables_dir / 'correlations.csv'}")
    print("\nCorrelations:")
    print(corr_df.to_string(index=False))
    
    # Descriptive stats by category
    stats_df = descriptive_stats_by_category(df, geo_features + ['is_hallucinated'])
    stats_df.to_csv(config.tables_dir / "stats_by_category.csv", index=False)
    print(f"\nSaved: {config.tables_dir / 'stats_by_category.csv'}")
    
    print("\nVisualization and analysis complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create visualizations")
    parser.add_argument("--config", default="experiments/config_example.yaml")
    args = parser.parse_args()
    
    create_visualizations(args.config)
