"""Within-category geometry analysis.

This module tests H1: Within each category, geometric features predict 
variation in hallucination probability.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.seed import set_seed
from src.utils.logging_utils import setup_logger


def analyze_category(df_category, category_name, output_dir, seed=42):
    """Analyze geometry-hallucination relationship within a single category."""
    logger = setup_logger(f"within_category_{category_name}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing category: {category_name}")
    logger.info(f"{'='*60}")
    
    # Check if we have enough samples
    if len(df_category) < 20:
        logger.warning(f"Too few samples ({len(df_category)}) for reliable analysis")
        return None
    
    # Check class balance
    n_hallucinated = df_category['is_hallucinated'].sum()
    n_correct = len(df_category) - n_hallucinated
    logger.info(f"Samples: {len(df_category)} ({n_hallucinated} hallucinated, {n_correct} correct)")
    
    if n_hallucinated < 5 or n_correct < 5:
        logger.warning(f"Insufficient samples in one class for {category_name}")
        return None
    
    # Prepare features
    geometry_features = ['local_id', 'curvature_score', 'oppositeness_score', 
                         'density', 'centrality']
    
    # Check which features are available
    available_features = [f for f in geometry_features if f in df_category.columns]
    
    if len(available_features) == 0:
        logger.error("No geometry features found!")
        return None
    
    logger.info(f"Using features: {available_features}")
    
    X = df_category[available_features].values
    y = df_category['is_hallucinated'].values
    
    # Handle missing values
    if np.isnan(X).any():
        logger.warning("Found NaN values, filling with column means")
        col_means = np.nanmean(X, axis=0)
        for i, mean_val in enumerate(col_means):
            X[:, i] = np.where(np.isnan(X[:, i]), mean_val, X[:, i])
    
    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )
    except ValueError:
        # If stratification fails, try without it
        logger.warning("Stratification failed, splitting without stratification")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed
        )
    
    results = {
        'category': category_name,
        'n_samples': len(df_category),
        'n_hallucinated': int(n_hallucinated),
        'hallucination_rate': float(n_hallucinated / len(df_category))
    }
    
    # Train Logistic Regression
    logger.info("\nTraining Logistic Regression...")
    lr = LogisticRegression(random_state=seed, max_iter=1000)
    lr.fit(X_train, y_train)
    
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    
    try:
        auc_lr = roc_auc_score(y_test, y_prob_lr)
    except ValueError:
        auc_lr = 0.5  # Only one class in test set
    
    acc_lr = accuracy_score(y_test, y_pred_lr)
    
    results['lr_auc'] = float(auc_lr)
    results['lr_accuracy'] = float(acc_lr)
    results['lr_coefficients'] = {
        feat: float(coef) 
        for feat, coef in zip(available_features, lr.coef_[0])
    }
    
    logger.info(f"Logistic Regression - AUC: {auc_lr:.3f}, Accuracy: {acc_lr:.3f}")
    logger.info("Feature coefficients:")
    for feat, coef in results['lr_coefficients'].items():
        logger.info(f"  {feat}: {coef:+.4f}")
    
    # Train Random Forest
    logger.info("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, max_depth=5)
    rf.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    
    try:
        auc_rf = roc_auc_score(y_test, y_prob_rf)
    except ValueError:
        auc_rf = 0.5
    
    acc_rf = accuracy_score(y_test, y_pred_rf)
    
    results['rf_auc'] = float(auc_rf)
    results['rf_accuracy'] = float(acc_rf)
    results['rf_feature_importance'] = {
        feat: float(imp) 
        for feat, imp in zip(available_features, rf.feature_importances_)
    }
    
    logger.info(f"Random Forest - AUC: {auc_rf:.3f}, Accuracy: {acc_rf:.3f}")
    logger.info("Feature importance:")
    for feat, imp in results['rf_feature_importance'].items():
        logger.info(f"  {feat}: {imp:.4f}")
    
    # Feature statistics by outcome
    logger.info("\nGeometry by outcome:")
    hall_mask = df_category['is_hallucinated'] == 1
    for feat in available_features:
        if feat in df_category.columns:
            hall_mean = df_category[hall_mask][feat].mean()
            correct_mean = df_category[~hall_mask][feat].mean()
            logger.info(f"  {feat}: Hallucinated={hall_mean:.4f}, Correct={correct_mean:.4f}")
    
    return results


def main():
    """Run within-category analysis on V2 results."""
    set_seed(42)
    
    # Load V2 results
    results_file = Path("results/all_results.csv")
    if not results_file.exists():
        print(f"Error: {results_file} not found. Please run V2 pipeline first.")
        return
    
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} samples from V2 results")
    
    # Create output directory
    output_dir = Path("results/v3")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze each category
    all_results = []
    categories = df['category'].unique()
    
    for category in categories:
        df_cat = df[df['category'] == category]
        result = analyze_category(df_cat, category, output_dir, seed=42)
        if result:
            all_results.append(result)
    
    # Save results
    output_file = output_dir / "within_category_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Within-category analysis complete!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    # Create summary table
    summary_rows = []
    for result in all_results:
        summary_rows.append({
            'category': result['category'],
            'n_samples': result['n_samples'],
            'hallucination_rate': f"{result['hallucination_rate']:.1%}",
            'lr_auc': f"{result['lr_auc']:.3f}",
            'rf_auc': f"{result['rf_auc']:.3f}",
            'top_lr_feature': max(result['lr_coefficients'].items(), 
                                  key=lambda x: abs(x[1]))[0],
            'top_rf_feature': max(result['rf_feature_importance'].items(), 
                                  key=lambda x: x[1])[0]
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_file = output_dir / "within_category_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary table saved to: {summary_file}")
    print("\n" + summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
