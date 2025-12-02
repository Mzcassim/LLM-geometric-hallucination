"""Robustness analysis across different embedding models.

Compares geometry-hallucination correlations using:
1. Original: text-embedding-3-small (1536-dim)
2. Alternative 1: text-embedding-3-large (3072-dim)
3. Alternative 2: all-mpnet-base-v2 (768-dim, open source)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.geometry.compute_features import compute_geometry_features


def load_embeddings(embedding_file):
    """Load embedding file."""
    return np.load(embedding_file)


def compute_correlation(results_df, geometry_feature='curvature_score'):
    """Compute Spearman correlation between geometry and hallucination."""
    correlation, p_value = stats.spearmanr(
        results_df[geometry_feature],
        results_df['is_hallucinated']
    )
    return correlation, p_value


def analyze_robustness(original_results, alternative_embeddings_dir, output_dir):
    """Compare correlations across embedding models."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load original results (from V2 or V3)
    df = pd.read_csv(original_results)
    
    # Models to test
    embedding_models = [
        {
            'name': 'text-embedding-3-small (Original)',
            'file': 'data/processed/question_embeddings.npy',
            'color': 'blue'
        },
        {
            'name': 'text-embedding-3-large',
            'file': f'{alternative_embeddings_dir}/embeddings_openai_large.npy',
            'color': 'green'
        },
        {
            'name': 'all-mpnet-base-v2 (Open Source)',
            'file': f'{alternative_embeddings_dir}/embeddings_mpnet.npy',
            'color': 'orange'
        }
    ]
    
    results = []
    
    for model_info in embedding_models:
        print(f"\nAnalyzing: {model_info['name']}")
        
        if not Path(model_info['file']).exists():
            print(f"  Skipping (file not found: {model_info['file']})")
            continue
        
        # Load embeddings
        embeddings = load_embeddings(model_info['file'])
        print(f"  Loaded embeddings: {embeddings.shape}")
        
        # Compute geometry
        print("  Computing geometry features...")
        geometry_df = compute_geometry_features(embeddings)
        
        # Merge with hallucination labels
        merged_df = df.merge(geometry_df, left_on='id', right_index=True, how='left')
        
        # Compute correlations for each feature
        for feature in ['curvature_score', 'density', 'centrality']:
            if feature in merged_df.columns:
                corr, p_val = compute_correlation(merged_df, feature)
                
                results.append({
                    'embedding_model': model_info['name'],
                    'geometry_feature': feature,
                    'correlation': corr,
                    'p_value': p_val
                })
                
                print(f"    {feature}: r={corr:.3f}, p={p_val:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/embedding_robustness_results.csv", index=False)
    
    # Create visualization
    pivot = results_df.pivot(index='geometry_feature', columns='embedding_model', values='correlation')
    
    plt.figure(figsize=(10, 6))
    pivot.plot(kind='bar', rot=0)
    plt.title('Geometry-Hallucination Correlation Across Embedding Models', fontsize=14, fontweight='bold')
    plt.ylabel('Spearman Correlation (r)', fontsize=12)
    plt.xlabel('Geometry Feature', fontsize=12)
    plt.legend(title='Embedding Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/embedding_robustness_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Robustness analysis complete!")
    print(f"Results saved to: {output_dir}/")
    
    return results_df


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", default="results/v3/multi_model/all_models_results.csv")
    parser.add_argument("--alt-embeddings-dir", default="data/processed/alternative_embeddings")
    parser.add_argument("--output-dir", default="results/v3/robustness")
    
    args = parser.parse_args()
    
    analyze_robustness(args.results_file, args.alt_embeddings_dir, args.output_dir)


if __name__ == "__main__":
    main()
