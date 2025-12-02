"""Aggregate multi-model results with geometric features.

Merges judged answers from multiple models with the pre-computed geometric features
to enable full V3 analysis (visualizations, statistical tests) for all models.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.io import read_jsonl


def load_geometry_features():
    """Load pre-computed geometric features."""
    # We can load from the V2 results which have everything
    v2_results = pd.read_csv("archive/v2_production_run/results/all_results.csv")
    
    # Create a mapping from question ID to geometry features
    geometry_cols = ['id', 'local_id', 'curvature_score', 'oppositeness_score', 'density', 'centrality']
    
    # Check which columns exist
    available_cols = [c for c in geometry_cols if c in v2_results.columns]
    
    geometry_df = v2_results[available_cols].copy()
    return geometry_df


def aggregate_results(input_dir, output_file):
    """Aggregate multi-model results and merge with geometry."""
    
    input_path = Path(input_dir)
    files = list(input_path.glob("judged_answers_*.jsonl"))
    
    if not files:
        print(f"No judged files found in {input_dir}")
        return
    
    print(f"Found {len(files)} judged files")
    
    # Load geometry
    geometry_df = load_geometry_features()
    print(f"Loaded geometry for {len(geometry_df)} prompts")
    
    all_data = []
    
    for file in files:
        model_name = file.stem.replace("judged_answers_", "")
        print(f"Processing {model_name}...")
        
        data = read_jsonl(file)
        model_df = pd.DataFrame(data)
        
        # Ensure ID column exists
        if 'id' not in model_df.columns:
            print(f"  Warning: No 'id' column in {file.name}")
            continue
            
        # Merge with geometry
        merged_df = model_df.merge(geometry_df, on='id', how='left')
        
        # Add model name
        merged_df['model_name'] = model_name
        
        # Standardize columns for analysis scripts
        merged_df['is_hallucinated'] = (merged_df['judge_label'] == 2).astype(int)
        
        all_data.append(merged_df)
    
    # Combine all
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Save
    final_df.to_csv(output_file, index=False)
    print(f"\nSaved aggregated results to {output_file}")
    print(f"Total samples: {len(final_df)}")
    print(f"Models included: {final_df['model_name'].unique()}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate multi-model results")
    parser.add_argument("--input-dir", type=str, default="results/v3/multi_model/judged")
    parser.add_argument("--output-file", type=str, default="results/v3/multi_model/all_models_results.csv")
    
    args = parser.parse_args()
    
    aggregate_results(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
