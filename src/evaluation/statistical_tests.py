"""Statistical tests for multi-model hallucination analysis."""

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations


def kendall_tau_correlation(results_df):
    """
    Compute Kendall's Tau correlation between all model pairs.
    
    Args:
        results_df: DataFrame with columns ['id', 'model_name', 'is_hallucinated']
    
    Returns:
        DataFrame with correlation matrix
    """
    # Drop duplicates if any
    results_df = results_df.drop_duplicates(subset=['id', 'model_name'])
    
    # Pivot to get model columns
    pivot = results_df.pivot(index='id', columns='model_name', values='is_hallucinated')
    
    models = pivot.columns.tolist()
    n_models = len(models)
    
    # Initialize matrices
    tau_matrix = np.zeros((n_models, n_models))
    p_matrix = np.zeros((n_models, n_models))
    
    for i, model_a in enumerate(models):
        for j, model_b in enumerate(models):
            if i == j:
                tau_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
            else:
                # Remove NaN pairs
                mask = ~(pivot[model_a].isna() | pivot[model_b].isna())
                if mask.sum() > 0:
                    tau, p_value = stats.kendalltau(
                        pivot[model_a][mask],
                        pivot[model_b][mask]
                    )
                    tau_matrix[i, j] = tau
                    p_matrix[i, j] = p_value
    
    # Create DataFrames
    tau_df = pd.DataFrame(tau_matrix, index=models, columns=models)
    p_df = pd.DataFrame(p_matrix, index=models, columns=models)
    
    return tau_df, p_df


def paired_ttest(before, after):
    """
    Perform paired t-test for before/after comparison.
    
    Args:
        before: Array of values before intervention
        after: Array of values after intervention
    
    Returns:
        dict with t-statistic, p-value, effect size (Cohen's d)
    """
    t_stat, p_value = stats.ttest_rel(before, after)
    
    # Cohen's d for paired samples
    diff = before - after
    cohens_d = diff.mean() / diff.std()
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'mean_diff': diff.mean()
    }


def logistic_regression_stats(X, y):
    """
    Compute logistic regression with statsmodels for p-values.
    
    Args:
        X: Feature matrix (DataFrame)
        y: Target variable (hallucination labels)
    
    Returns:
        DataFrame with coefficients, p-values, odds ratios
    """
    try:
        import statsmodels.api as sm
        
        # Add constant
        X_with_const = sm.add_constant(X)
        
        # Fit model
        model = sm.Logit(y, X_with_const).fit(disp=0)
        
        # Extract results
        results = pd.DataFrame({
            'feature': X_with_const.columns,
            'coefficient': model.params,
            'std_error': model.bse,
            'z_score': model.tvalues,
            'p_value': model.pvalues,
            'odds_ratio': np.exp(model.params)
        })
        
        return results
        
    except ImportError:
        print("Warning: statsmodels not installed, skipping p-values")
        return None


def compute_agreement_rate(model_a_labels, model_b_labels):
    """Compute simple agreement rate between two models."""
    return (model_a_labels == model_b_labels).mean()


def main():
    """Run all statistical tests and save results."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, help="Path to all_models_results.csv")
    parser.add_argument("--output-dir", default="results/v3/multi_model/stats")
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.input_file)
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Computing Kendall's Tau correlation...")
    tau_df, p_df = kendall_tau_correlation(df)
    
    tau_df.to_csv(f"{args.output_dir}/kendall_tau_matrix.csv")
    p_df.to_csv(f"{args.output_dir}/kendall_tau_pvalues.csv")
    
    print(f"Saved to {args.output_dir}/kendall_tau_matrix.csv")
    
    # Compute logistic regression if geometry features exist
    if 'curvature_score' in df.columns:
        print("\nComputing logistic regression with p-values...")
        
        geometry_features = ['curvature_score', 'density', 'centrality']
        if 'local_id' in df.columns:
            geometry_features.append('local_id')
        
        available_features = [f for f in geometry_features if f in df.columns]
        
        if available_features:
            X = df[available_features].dropna()
            y = df.loc[X.index, 'is_hallucinated']
            
            results = logistic_regression_stats(X, y)
            if results is not None:
                results.to_csv(f"{args.output_dir}/logistic_regression_stats.csv", index=False)
                print(f"Saved to {args.output_dir}/logistic_regression_stats.csv")
    
    print("\nStatistical analysis complete!")


if __name__ == "__main__":
    main()
