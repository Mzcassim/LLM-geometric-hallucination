"""Generate publication-ready tables for the paper."""

import pandas as pd
import numpy as np


def generate_model_performance_table(results_df, output_file):
    """
    Generate Table 1: Model Performance Summary.
    
    Columns: Model, Total Prompts, Hallucinations, Rate, AUC (if available)
    """
    # Group by model
    summary = results_df.groupby('model_name').agg({
        'id': 'count',
        'is_hallucinated': ['sum', 'mean']
    }).round(4)
    
    summary.columns = ['Total_Prompts', 'Hallucinations', 'Hallucination_Rate']
    summary['Hallucination_Rate'] = (summary['Hallucination_Rate'] * 100).round(2)
    
    # Sort by hallucination rate
    summary = summary.sort_values('Hallucination_Rate')
    
    summary.to_csv(output_file)
    print(f"Table 1 saved to {output_file}")
    
    return summary


def generate_feature_importance_table(model_results_file, output_file):
    """
    Generate Table 2: Feature Importance.
    
    Uses predictive model results if available.
    """
    try:
        results = pd.read_csv(model_results_file)
        
        # Filter for Random Forest geometry-only model
        rf_results = results[
            (results['model'] == 'random_forest') & 
            (results['feature_set'] == 'geometry_only')
        ]
        
        if len(rf_results) > 0:
            # Extract feature importance (assuming it's stored in a JSON column or separate file)
            # This is a placeholder - actual implementation depends on how feature importance is stored
            print(f"Feature importance data found in {model_results_file}")
            print("Note: Actual extraction depends on data format")
        else:
            print("Random Forest results not found in predictive model output")
            
    except FileNotFoundError:
        print(f"File not found: {model_results_file}")
        print("Run predictive modeling first")


def generate_consistency_summary_table(tau_matrix_file, output_file):
    """
    Generate Table: Cross-Model Consistency.
    
    Summary statistics from Kendall's Tau matrix.
    """
    tau_df = pd.read_csv(tau_matrix_file, index_col=0)
    
    # Get upper triangle (excluding diagonal)
    models = tau_df.columns.tolist()
    tau_values = []
    
    for i, model_a in enumerate(models):
        for j, model_b in enumerate(models):
            if i < j:
                tau_values.append(tau_df.loc[model_a, model_b])
    
    summary = pd.DataFrame({
        'Metric': ['Mean Tau', 'Median Tau', 'Min Tau', 'Max Tau', 'Std Tau'],
        'Value': [
            np.mean(tau_values),
            np.median(tau_values),
            np.min(tau_values),
            np.max(tau_values),
            np.std(tau_values)
        ]
    }).round(3)
    
    summary.to_csv(output_file, index=False)
    print(f"Consistency summary saved to {output_file}")
    
    return summary


def generate_all_tables(results_file, stats_dir, output_dir):
    """Generate all tables at once."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load main results
    df = pd.read_csv(results_file)
    
    # Table 1: Model Performance
    print("\n=== TABLE 1: Model Performance ===")
    perf_table = generate_model_performance_table(
        df, 
        f"{output_dir}/table1_model_performance.csv"
    )
    print(perf_table)
    
    # Table 2: Consistency
    tau_file = f"{stats_dir}/kendall_tau_matrix.csv"
    if os.path.exists(tau_file):
        print("\n=== TABLE 2: Cross-Model Consistency ===")
        consistency_table = generate_consistency_summary_table(
            tau_file,
            f"{output_dir}/table2_consistency_summary.csv"
        )
        print(consistency_table)
    else:
        print(f"\nKendall's Tau matrix not found at {tau_file}")
        print("Run statistical_tests.py first")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", default="results/v3/multi_model/all_models_results.csv")
    parser.add_argument("--stats-dir", default="results/v3/multi_model/stats")
    parser.add_argument("--output-dir", default="results/v3/multi_model/tables")
    args = parser.parse_args()
    
    generate_all_tables(args.results_file, args.stats_dir, args.output_dir)
    
    print("\n✅ All tables generated!")
    print(f"Check {args.output_dir}/ for CSV files")


if __name__ == "__main__":
    main()
