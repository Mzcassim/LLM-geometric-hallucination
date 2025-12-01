"""Conditional model comparison: Category vs Category+Geometry.

Tests whether geometry provides incremental predictive value beyond category labels.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from scipy.stats import chi2
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.seed import set_seed


def prepare_features(df, feature_set='combined'):
    """Prepare feature matrix based on feature set type."""
    
    if feature_set == 'category_only':
        # One-hot encode categories
        cat_dummies = pd.get_dummies(df['category'], prefix='cat')
        X = cat_dummies.values
        feature_names = list(cat_dummies.columns)
        
    elif feature_set == 'geometry_only':
        # Geometry features
        geometry_features = ['local_id', 'curvature_score', 'oppositeness_score', 
                            'density', 'centrality']
        available = [f for f in geometry_features if f in df.columns]
        X = df[available].values
        feature_names = available
        
    elif feature_set == 'combined':
        # Both category and geometry
        cat_dummies = pd.get_dummies(df['category'], prefix='cat')
        geometry_features = ['local_id', 'curvature_score', 'oppositeness_score', 
                            'density', 'centrality']
        available_geo = [f for f in geometry_features if f in df.columns]
        
        X_cat = cat_dummies.values
        X_geo = df[available_geo].values
        X = np.hstack([X_cat, X_geo])
        feature_names = list(cat_dummies.columns) + available_geo
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    
    # Handle NaN values
    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)
        for i in range(X.shape[1]):
            X[:, i] = np.where(np.isnan(X[:, i]), col_means[i], X[:, i])
    
    return X, feature_names


def cross_val_comparison(df, n_folds=5, seed=42):
    """Compare models using stratified k-fold cross-validation."""
    
    print("Running cross-validation comparison...")
    print(f"Dataset size: {len(df)}, Hallucination rate: {df['is_hallucinated'].mean():.1%}\n")
    
    y = df['is_hallucinated'].values
    
    # Prepare all feature sets
    X_cat, feat_cat = prepare_features(df, 'category_only')
    X_geo, feat_geo = prepare_features(df, 'geometry_only')
    X_comb, feat_comb = prepare_features(df, 'combined')
    
    # Cross-validation setup
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    results = {
        'category_only': {'auc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
        'geometry_only': {'auc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
        'combined': {'auc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    }
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_cat, y)):
        print(f"Fold {fold_idx + 1}/{n_folds}...")
        
        # Category-only model
        lr_cat = LogisticRegression(random_state=seed, max_iter=1000)
        lr_cat.fit(X_cat[train_idx], y[train_idx])
        y_prob_cat = lr_cat.predict_proba(X_cat[test_idx])[:, 1]
        y_pred_cat = lr_cat.predict(X_cat[test_idx])
        
        results['category_only']['auc'].append(roc_auc_score(y[test_idx], y_prob_cat))
        results['category_only']['accuracy'].append(accuracy_score(y[test_idx], y_pred_cat))
        p, r, f, _ = precision_recall_fscore_support(y[test_idx], y_pred_cat, average='binary', zero_division=0)
        results['category_only']['precision'].append(p)
        results['category_only']['recall'].append(r)
        results['category_only']['f1'].append(f)
        
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
        lr_comb = LogisticRegression(random_state=seed, max_iter=1000)
        lr_comb.fit(X_comb[train_idx], y[train_idx])
        y_prob_comb = lr_comb.predict_proba(X_comb[test_idx])[:, 1]
        y_pred_comb = lr_comb.predict(X_comb[test_idx])
        
        results['combined']['auc'].append(roc_auc_score(y[test_idx], y_prob_comb))
        results['combined']['accuracy'].append(accuracy_score(y[test_idx], y_pred_comb))
        p, r, f, _ = precision_recall_fscore_support(y[test_idx], y_pred_comb, average='binary', zero_division=0)
        results['combined']['precision'].append(p)
        results['combined']['recall'].append(r)
        results['combined']['f1'].append(f)
    
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


def likelihood_ratio_test(df, seed=42):
    """Perform likelihood-ratio test for nested models."""
    
    print("\nPerforming likelihood-ratio test...")
    
    y = df['is_hallucinated'].values
    X_cat, _ = prepare_features(df, 'category_only')
    X_comb, _ = prepare_features(df, 'combined')
    
    # Train both models on full dataset
    lr_cat = LogisticRegression(random_state=seed, max_iter=1000)
    lr_cat.fit(X_cat, y)
    
    lr_comb = LogisticRegression(random_state=seed, max_iter=1000)
    lr_comb.fit(X_comb, y)
    
    # Compute log-likelihoods
    from sklearn.metrics import log_loss
    
    # Negative log-likelihood
    nll_cat = log_loss(y, lr_cat.predict_proba(X_cat)[:, 1], normalize=False)
    nll_comb = log_loss(y, lr_comb.predict_proba(X_comb)[:, 1], normalize=False)
    
    # Likelihood ratio test statistic
    # LR = 2 * (log L_full - log L_reduced)
    lr_stat = 2 * (nll_cat - nll_comb)
    
    # Degrees of freedom = difference in number of parameters
    df_cat = X_cat.shape[1]
    df_comb = X_comb.shape[1]
    df_diff = df_comb - df_cat
    
    # P-value from chi-squared distribution
    p_value = 1 - chi2.cdf(lr_stat, df_diff)
    
    print(f"  Category-only: {df_cat} parameters, NLL = {nll_cat:.2f}")
    print(f"  Combined: {df_comb} parameters, NLL = {nll_comb:.2f}")
    print(f"  LR statistic: {lr_stat:.4f}")
    print(f"  Degrees of freedom: {df_diff}")
    print(f"  P-value: {p_value:.4e}")
    
    if p_value < 0.05:
        print("  ✓ Geometry provides significant improvement (p < 0.05)")
    else:
        print("  ✗ No significant improvement from geometry")
    
    return {
        'lr_statistic': float(lr_stat),
        'df': int(df_diff),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05)
    }


def main():
    """Run conditional model comparison."""
    set_seed(42)
    
    # Load V2 results
    results_file = Path("results/all_results.csv")
    if not results_file.exists():
        print(f"Error: {results_file} not found.")
        return
    
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} samples\n")
    print("="*60)
    
    # Cross-validation comparison
    cv_results = cross_val_comparison(df, n_folds=5, seed=42)
    
    # Print summary
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS (5-fold)")
    print("="*60)
    for model_name, metrics in cv_results.items():
        print(f"\n{model_name.upper()}:")
        for metric, stats in metrics.items():
            print(f"  {metric:12s}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    # Likelihood-ratio test
    lr_test = likelihood_ratio_test(df, seed=42)
    
    # Save results
    output_dir = Path("results/v3")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        'cross_validation': cv_results,
        'likelihood_ratio_test': lr_test
    }
    
    output_file = output_dir / "conditional_model_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print("="*60)


if __name__ == "__main__":
    main()
