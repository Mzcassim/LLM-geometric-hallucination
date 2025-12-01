"""Text vs Geometry model comparison.

Compares models using textual features, geometric features, and both,
to show that geometry adds incremental value beyond simple lexical cues.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.seed import set_seed


def prepare_features(df, feature_set='combined'):
    """Prepare feature matrix based on feature set type."""
    
    if feature_set == 'text_only':
        # Textual features
        text_features = ['n_tokens', 'n_chars', 'lexical_diversity', 'avg_word_len',
                        'trigram_entropy', 'capital_ratio', 'has_numbers', 
                        'n_uncertainty_markers', 'n_academic_words', 'n_fiction_words']
        available = [f for f in text_features if f in df.columns]
        X = df[available].values
        feature_names = available
        
    elif feature_set == 'geometry_only':
        # Geometry features
        geometry_features = ['local_id', 'curvature_score', 'oppositeness_score', 
                            'density', 'centrality']
        available = [f for f in geometry_features if f in df.columns]
        X = df[available].values
        feature_names = available
        
    elif feature_set == 'text_and_geometry':
        # Both text and geometry
        text_features = ['n_tokens', 'n_chars', 'lexical_diversity', 'avg_word_len',
                        'trigram_entropy', 'capital_ratio', 'has_numbers',
                        'n_uncertainty_markers', 'n_academic_words', 'n_fiction_words']
        geometry_features = ['local_id', 'curvature_score', 'oppositeness_score', 
                            'density', 'centrality']
        
        available_text = [f for f in text_features if f in df.columns]
        available_geo = [f for f in geometry_features if f in df.columns]
        
        X_text = df[available_text].values
        X_geo = df[available_geo].values
        X = np.hstack([X_text, X_geo])
        feature_names = available_text + available_geo
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    
    # Handle NaN values
    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)
        for i in range(X.shape[1]):
            X[:, i] = np.where(np.isnan(X[:, i]), col_means[i], X[:, i])
    
    return X, feature_names


def cross_val_comparison(df, n_folds=5, seed=42):
    """Compare text vs geometry vs combined using cross-validation."""
    
    print("Running cross-validation comparison...")
    print(f"Dataset size: {len(df)}, Hallucination rate: {df['is_hallucinated'].mean():.1%}\n")
    
    y = df['is_hallucinated'].values
    
    # Prepare all feature sets
    X_text, feat_text = prepare_features(df, 'text_only')
    X_geo, feat_geo = prepare_features(df, 'geometry_only')
    X_both, feat_both = prepare_features(df, 'text_and_geometry')
    
    print(f"Text features: {len(feat_text)}")
    print(f"Geometry features: {len(feat_geo)}")
    print(f"Combined features: {len(feat_both)}\n")
    
    # Cross-validation setup
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    results = {
        'text_only': {'auc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
        'geometry_only': {'auc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
        'text_and_geometry': {'auc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    }
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_text, y)):
        print(f"Fold {fold_idx + 1}/{n_folds}...")
        
        # Text-only model
        lr_text = LogisticRegression(random_state=seed, max_iter=1000)
        lr_text.fit(X_text[train_idx], y[train_idx])
        y_prob_text = lr_text.predict_proba(X_text[test_idx])[:, 1]
        y_pred_text = lr_text.predict(X_text[test_idx])
        
        results['text_only']['auc'].append(roc_auc_score(y[test_idx], y_prob_text))
        results['text_only']['accuracy'].append(accuracy_score(y[test_idx], y_pred_text))
        p, r, f, _ = precision_recall_fscore_support(y[test_idx], y_pred_text, average='binary', zero_division=0)
        results['text_only']['precision'].append(p)
        results['text_only']['recall'].append(r)
        results['text_only']['f1'].append(f)
        
        # Geometry-only model
        lr_geo = LogisticRegression(random_state=seed, max_iter=1000)
        lr_geo.fit(X_geo[train_idx], y[train_idx])
        y_prob_geo = lr_geo.predict_proba(X_geo[test_idx])[:, 1]
        y_pred_geo = lr_geo.predict(X_geo[test_idx])
        
        results['geometry_only']['auc'].append(roc_auc_score(y[test_idx], y_prob_geo))
        results['geometry_only']['accuracy'].append(accuracy_score(y[test_idx], y_pred_geo))
        p, r, f, _ = precision_recall_fscore_support(y[test_idx], y_pred_geo, average='binary', zero_division=0)
        results['geometry_only']['precision'].append(p)
        results['geometry_only']['recall'].append(r)
        results['geometry_only']['f1'].append(f)
        
        # Combined model
        lr_both = LogisticRegression(random_state=seed, max_iter=1000)
        lr_both.fit(X_both[train_idx], y[train_idx])
        y_prob_both = lr_both.predict_proba(X_both[test_idx])[:, 1]
        y_pred_both = lr_both.predict(X_both[test_idx])
        
        results['text_and_geometry']['auc'].append(roc_auc_score(y[test_idx], y_prob_both))
        results['text_and_geometry']['accuracy'].append(accuracy_score(y[test_idx], y_pred_both))
        p, r, f, _ = precision_recall_fscore_support(y[test_idx], y_pred_both, average='binary', zero_division=0)
        results['text_and_geometry']['precision'].append(p)
        results['text_and_geometry']['recall'].append(r)
        results['text_and_geometry']['f1'].append(f)
    
    # Compute means and stds
    summary = {}
    for model_name, metrics in results.items():
        summary[model_name] = {
            metric: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
            for metric, values in metrics.items()
        }
    
    return summary


def main():
    """Run text vs geometry comparison."""
    
    set_seed(42)
    
    # Load merged data (results + textual features)
    results_file = Path("results/all_results.csv")
    text_features_file = Path("data/processed/textual_features.csv")
    
    if not results_file.exists():
        print(f"Error: {results_file} not found")
        return
    
    if not text_features_file.exists():
        print(f"Error: {text_features_file} not found")
        print("Please run: python -m src.features.textual_features")
        return
    
    # Load and merge
    df_results = pd.read_csv(results_file)
    df_text = pd.read_csv(text_features_file)
    
    # Merge on id
    df = df_results.merge(df_text, on='id', how='left')
    
    print(f"Loaded {len(df)} samples\n")
    print("="*60)
    
    # Cross-validation comparison
    cv_results = cross_val_comparison(df, n_folds=5, seed=42)
    
    # Print summary
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS (5-fold)")
    print("="*60)
    for model_name, metrics in cv_results.items():
        print(f"\n{model_name.upper().replace('_', ' ')}:")
        for metric, stats in metrics.items():
            print(f"  {metric:12s}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    # Save results
    output_dir = Path("results/v3")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "text_vs_geometry_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print("="*60)
    
    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    text_auc = cv_results['text_only']['auc']['mean']
    geo_auc = cv_results['geometry_only']['auc']['mean']
    both_auc = cv_results['text_and_geometry']['auc']['mean']
    
    print(f"\nText-only AUC: {text_auc:.3f}")
    print(f"Geometry-only AUC: {geo_auc:.3f}")
    print(f"Text+Geometry AUC: {both_auc:.3f}")
    
    if both_auc > text_auc:
        improvement = both_auc - text_auc
        print(f"\n✓ Geometry adds {improvement:+.3f} AUC beyond text features")
        print("  → Geometry is NOT just re-encoding lexical cues!")
    else:
        print("\n✗ Geometry does not improve beyond text features")
        print("  → Geometry might be encoding lexical properties")


if __name__ == "__main__":
    main()
