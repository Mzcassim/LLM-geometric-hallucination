"""Evaluation metrics for analyzing geometry-hallucination relationships."""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Tuple


def compute_correlations(
    df: pd.DataFrame,
    geometry_features: list[str],
    target: str = 'is_hallucinated'
) -> pd.DataFrame:
    """
    Compute correlations between geometry features and hallucination.
    
    Args:
        df: DataFrame with geometry features and hallucination labels
        geometry_features: List of geometry feature column names
        target: Target variable (default: 'is_hallucinated')
        
    Returns:
        DataFrame with correlation results
    """
    results = []
    
    for feature in geometry_features:
        # Remove NaN values for correlation
        valid_mask = ~(df[feature].isna() | df[target].isna())
        valid_df = df[valid_mask]
        
        if len(valid_df) < 2:
            results.append({
                'feature': feature,
                'pearson_r': np.nan,
                'pearson_p': np.nan,
                'spearman_r': np.nan,
                'spearman_p': np.nan,
                'n_samples': len(valid_df)
            })
            continue
        
        # Pearson correlation
        pearson_r, pearson_p = pearsonr(valid_df[feature], valid_df[target])
        
        # Spearman correlation
        spearman_r, spearman_p = spearmanr(valid_df[feature], valid_df[target])
        
        results.append({
            'feature': feature,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'n_samples': len(valid_df)
        })
    
    return pd.DataFrame(results)


def descriptive_stats_by_category(
    df: pd.DataFrame,
    features: list[str],
    category_col: str = 'category'
) -> pd.DataFrame:
    """
    Compute descriptive statistics for features by category.
    
    Args:
        df: DataFrame with features and categories
        features: List of feature column names
        category_col: Name of category column
        
    Returns:
        DataFrame with statistics by category
    """
    results = []
    
    for category in df[category_col].unique():
        cat_df = df[df[category_col] == category]
        
        row = {'category': category, 'n_samples': len(cat_df)}
        
        for feature in features:
            valid_values = cat_df[feature].dropna()
            if len(valid_values) > 0:
                row[f'{feature}_mean'] = valid_values.mean()
                row[f'{feature}_std'] = valid_values.std()
                row[f'{feature}_median'] = valid_values.median()
            else:
                row[f'{feature}_mean'] = np.nan
                row[f'{feature}_std'] = np.nan
                row[f'{feature}_median'] = np.nan
        
        results.append(row)
    
    return pd.DataFrame(results)


def compute_hallucination_rates_by_bins(
    df: pd.DataFrame,
    feature: str,
    n_bins: int = 5,
    target: str = 'is_hallucinated'
) -> pd.DataFrame:
    """
    Compute hallucination rates by binned feature values.
    
    Args:
        df: DataFrame with features and labels
        feature: Feature to bin
        n_bins: Number of bins
        target: Target variable
        
    Returns:
        DataFrame with hallucination rates per bin
    """
    # Remove NaN values
    valid_df = df[[feature, target]].dropna()
    
    if len(valid_df) < n_bins:
        return pd.DataFrame()
    
    # Create bins
    valid_df = valid_df.copy()
    valid_df['bin'] = pd.qcut(valid_df[feature], q=n_bins, labels=False, duplicates='drop')
    
    # Compute rates per bin
    results = []
    for bin_idx in sorted(valid_df['bin'].unique()):
        bin_df = valid_df[valid_df['bin'] == bin_idx]
        
        results.append({
            'bin': bin_idx,
            'bin_min': bin_df[feature].min(),
            'bin_max': bin_df[feature].max(),
            'bin_mean': bin_df[feature].mean(),
            'n_samples': len(bin_df),
            'hallucination_rate': bin_df[target].mean(),
            'n_hallucinated': bin_df[target].sum()
        })
    
    return pd.DataFrame(results)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        
    Returns:
        Dictionary of metrics
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }
