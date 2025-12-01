"""Deep dive into rare factual hallucinations.

Analyzes the 5% of factual questions that led to hallucinations, 
examining their geometric properties.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def analyze_factual_failures(df):
    """Analyze factual questions that hallucinated vs those that didn't."""
    
    # Filter to factual category
    df_factual = df[df['category'] == 'factual'].copy()
    
    print("="*60)
    print("FACTUAL CATEGORY ANALYSIS")
    print("="*60)
    print(f"Total factual questions: {len(df_factual)}")
    
    n_hall = df_factual['is_hallucinated'].sum()
    n_correct = len(df_factual) - n_hall
    
    print(f"Hallucinated: {n_hall} ({n_hall/len(df_factual)*100:.1f}%)")
    print(f"Correct: {n_correct} ({n_correct/len(df_factual)*100:.1f}%)")
    
    if n_hall < 2:
        print("\nInsufficient hallucinated samples for statistical analysis")
        return None
    
    # Split into hallucinated vs correct
    hall_mask = df_factual['is_hallucinated'] == 1
    df_hall = df_factual[hall_mask]
    df_correct = df_factual[~hall_mask]
    
    # Geometry features
    geometry_features = ['local_id', 'curvature_score', 'oppositeness_score', 
                         'density', 'centrality']
    available_features = [f for f in geometry_features if f in df_factual.columns]
    
    print(f"\nAnalyzing {len(available_features)} geometric features...\n")
    
    # Statistical comparison
    results = {
        'n_factual': len(df_factual),
        'n_hallucinated': int(n_hall),
        'n_correct': int(n_correct),
        'hallucination_rate': float(n_hall / len(df_factual)),
        'features': {}
    }
    
    print(f"{'Feature':<20} {'Hall Mean':<12} {'Correct Mean':<12} {'t-stat':<10} {'p-value':<10} {'Effect Size'}")
    print("-" * 90)
    
    for feat in available_features:
        hall_vals = df_hall[feat].dropna()
        correct_vals = df_correct[feat].dropna()
        
        if len(hall_vals) < 2 or len(correct_vals) < 2:
            continue
        
        hall_mean = hall_vals.mean()
        correct_mean = correct_vals.mean()
        
        # T-test
        t_stat, p_value = stats.ttest_ind(hall_vals, correct_vals)
        
        # Cohen's d (effect size)
        pooled_std = np.sqrt(((len(hall_vals)-1)*hall_vals.std()**2 + 
                              (len(correct_vals)-1)*correct_vals.std()**2) / 
                             (len(hall_vals) + len(correct_vals) - 2))
        cohens_d = (hall_mean - correct_mean) / pooled_std if pooled_std > 0 else 0
        
        print(f"{feat:<20} {hall_mean:<12.4f} {correct_mean:<12.4f} {t_stat:<10.3f} {p_value:<10.4f} {cohens_d:+.3f}")
        
        results['features'][feat] = {
            'hallucinated_mean': float(hall_mean),
            'correct_mean': float(correct_mean),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant': bool(p_value < 0.05)
        }
    
    # List the hallucinated factual questions
    print("\n" + "="*60)
    print("FACTUAL QUESTIONS THAT HALLUCINATED:")
    print("="*60)
    
    hallucinated_questions = []
    for idx, row in df_hall.iterrows():
        q = row['question']
        answer = row.get('model_answer', 'N/A')
        
        print(f"\n{len(hallucinated_questions) + 1}. {q}")
        if 'judge_justification' in row:
            print(f"   Judge: {row['judge_justification']}")
        
        # Geometry profile
        geo_profile = {feat: float(row[feat]) for feat in available_features if feat in row}
        print(f"   Geometry: {geo_profile}")
        
        hallucinated_questions.append({
            'question': q,
            'answer': answer if pd.notna(answer) else 'N/A',
            'geometry': geo_profile
        })
    
    results['hallucinated_questions'] = hallucinated_questions
    
    return results


def create_visualization(df, output_dir):
    """Create violin plots comparing geometry distributions."""
    
    df_factual = df[df['category'] == 'factual'].copy()
    
    if df_factual['is_hallucinated'].sum() < 2:
        print("Skipping visualization (insufficient samples)")
        return
    
    geometry_features = ['local_id', 'curvature_score', 'oppositeness_score', 
                         'density', 'centrality']
    available_features = [f for f in geometry_features if f in df_factual.columns]
    
    fig, axes = plt.subplots(1, len(available_features), figsize=(4*len(available_features), 4))
    
    if len(available_features) == 1:
        axes = [axes]
    
    for ax, feat in zip(axes, available_features):
        hall_vals = df_factual[df_factual['is_hallucinated'] == 1][feat].dropna()
        correct_vals = df_factual[df_factual['is_hallucinated'] == 0][feat].dropna()
        
        parts = ax.violinplot([correct_vals, hall_vals], positions=[0, 1], 
                               showmeans=True, showmedians=True)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Correct', 'Hallucinated'])
        ax.set_ylabel(feat.replace('_', ' ').title())
        ax.set_title(f'{feat.replace("_", " ").title()}')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / "factual_failures_geometry.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_file}")


def main():
    """Run factual failures analysis."""
    
    # Load V2 results
    results_file = Path("results/all_results.csv")
    if not results_file.exists():
        print(f"Error: {results_file} not found.")
        return
    
    df = pd.read_csv(results_file)
    
    # Analyze
    results = analyze_factual_failures(df)
    
    if results is None:
        return
    
    # Save results
    output_dir = Path("results/v3")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "factual_failures_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualization(df, output_dir)
    
    print("\nFactual failures analysis complete!")


if __name__ == "__main__":
    main()
