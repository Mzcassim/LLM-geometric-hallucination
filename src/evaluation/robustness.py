"""Robustness checks and sensitivity analysis."""

import pandas as pd
import numpy as np
from typing import Callable, Dict


def filter_by_confidence(
    df: pd.DataFrame,
    min_confidence: float = 0.7,
    confidence_col: str = 'judge_confidence'
) -> pd.DataFrame:
    """
    Filter dataset to only high-confidence judgments.
    
    Args:
        df: DataFrame with confidence scores
        min_confidence: Minimum confidence threshold
        confidence_col: Name of confidence column
        
    Returns:
        Filtered DataFrame
    """
    return df[df[confidence_col] >= min_confidence].copy()


def stratified_analysis(
    df: pd.DataFrame,
    stratify_by: str,
    analysis_fn: Callable[[pd.DataFrame], Dict]
) -> pd.DataFrame:
    """
    Run analysis separately for each stratum.
    
    Args:
        df: DataFrame to analyze
        stratify_by: Column to stratify by
        analysis_fn: Function that takes a DataFrame and returns a dict of results
        
    Returns:
        DataFrame with results per stratum
    """
    results = []
    
    for stratum_value in df[stratify_by].unique():
        stratum_df = df[df[stratify_by] == stratum_value]
        
        try:
            stratum_results = analysis_fn(stratum_df)
            stratum_results[stratify_by] = stratum_value
            stratum_results['n_samples'] = len(stratum_df)
            results.append(stratum_results)
        except Exception as e:
            print(f"Error in stratum {stratum_value}: {e}")
    
    return pd.DataFrame(results)


def bootstrap_correlation(
    df: pd.DataFrame,
    feature: str,
    target: str,
    n_iterations: int = 1000,
    sample_frac: float = 0.8,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Compute bootstrap confidence intervals for correlation.
    
    Args:
        df: DataFrame with features
        feature: Feature column name
        target: Target column name
        n_iterations: Number of bootstrap iterations
        sample_frac: Fraction of data to sample each iteration
        random_state: Random seed
        
    Returns:
        Dictionary with correlation estimate and confidence intervals
    """
    from scipy.stats import spearmanr
    
    np.random.seed(random_state)
    
    # Remove NaN values
    valid_df = df[[feature, target]].dropna()
    
    if len(valid_df) < 10:
        return {
            'correlation': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan
        }
    
    correlations = []
    
    for _ in range(n_iterations):
        # Bootstrap sample
        sample = valid_df.sample(frac=sample_frac, replace=True)
        
        # Compute correlation
        corr, _ = spearmanr(sample[feature], sample[target])
        correlations.append(corr)
    
    correlations = np.array(correlations)
    
    return {
        'correlation': np.median(correlations),
        'ci_lower': np.percentile(correlations, 2.5),
        'ci_upper': np.percentile(correlations, 97.5),
        'std': np.std(correlations)
    }


def cross_model_comparison(
    dfs: Dict[str, pd.DataFrame],
    metric_fn: Callable[[pd.DataFrame], Dict]
) -> pd.DataFrame:
    """
    Compare metrics across different models.
    
    Args:
        dfs: Dictionary mapping model names to their DataFrames
        metric_fn: Function to compute metrics from a DataFrame
        
    Returns:
        DataFrame comparing models
    """
    results = []
    
    for model_name, df in dfs.items():
        try:
            metrics = metric_fn(df)
            metrics['model'] = model_name
            results.append(metrics)
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
    
    return pd.DataFrame(results)
