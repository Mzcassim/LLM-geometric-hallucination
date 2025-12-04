
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.prediction import train_and_evaluate, HallucinationPredictor
from src.evaluation.early_warning import generate_early_warning_report

def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "results" / "v3" / "prediction"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_path = base_dir / "results" / "v3" / "multi_model" / "all_models_results.csv"
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")
    
    # Filter for samples with geometry
    df_geom = df.dropna(subset=['centrality', 'curvature_score', 'local_id'])
    print(f"Samples with geometry: {len(df_geom)}")
    
    # Define feature sets
    feature_sets = [
        {'name': 'Category Only', 'type': 'category_only'},
        {'name': 'Geometry Only', 'type': 'geometry_only'},
        {'name': 'Combined', 'type': 'combined'}
    ]
    
    # Run prediction analysis
    print("\nRunning prediction analysis...")
    results = train_and_evaluate(
        df_geom,
        feature_sets=feature_sets,
        model_types=["logistic", "random_forest"],
        output_dir=results_dir
    )
    
    # Run early warning analysis on best model
    best = results['best_model']
    print(f"\nBest model: {best['feature_set']} ({best['model_type']})")
    print(f"Test AUC: {best['test_auc']:.3f}")
    
    # We need to reconstruct the test set to run early warning analysis
    # This is a bit hacky but ensures we use the same split
    from sklearn.model_selection import train_test_split
    
    y = df_geom['is_hallucinated'].values
    indices = np.arange(len(df_geom))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.3, random_state=42, stratify=y
    )
    
    df_test = df_geom.iloc[test_idx]
    y_test = y[test_idx]
    
    predictor = best['predictor']
    X_test, _ = predictor.prepare_features(df_test, feature_set=best['feature_set'].lower().replace(' ', '_'))
    
    # Generate early warning report
    ew_dir = results_dir / "early_warning"
    generate_early_warning_report(
        predictor,
        X_test,
        y_test,
        df_test,
        output_dir=ew_dir
    )

if __name__ == "__main__":
    main()
