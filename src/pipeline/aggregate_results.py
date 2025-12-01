"""Aggregate all results into a single dataset."""

import sys
from pathlib import Path
import argparse
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import load_config, ProjectConfig
from src.utils.io import read_jsonl
from src.utils.logging_utils import setup_logger


def aggregate_results(config: ProjectConfig):
    """Aggregate judgments and geometry features."""
    logger = setup_logger(
        "aggregation",
        log_file=config.data_dir / "logs" / "aggregation.log"
    )
    
    logger.info("Starting result aggregation...")
    
    # Load judged answers
    judged_file = config.processed_dir / "judged_answers.jsonl"
    if not judged_file.exists():
        logger.error(f"Judged answers file not found: {judged_file}")
        logger.error("Please run run_judging.py first")
        return
    
    judged_data = read_jsonl(judged_file)
    judged_df = pd.DataFrame(judged_data)
    logger.info(f"Loaded {len(judged_df)} judged answers")
    
    # Load geometry features
    geometry_file = config.processed_dir / "geometry_features.csv"
    if not geometry_file.exists():
        logger.error(f"Geometry features file not found: {geometry_file}")
        logger.error("Please run compute_geometry.py first")
        return
    
    geometry_df = pd.read_csv(geometry_file)
    logger.info(f"Loaded {len(geometry_df)} geometry feature rows")
    
    # Merge on ID
    merged_df = judged_df.merge(geometry_df, on='id', how='left', suffixes=('', '_geo'))
    logger.info(f"Merged dataset has {len(merged_df)} rows")
    
    # Create derived features
    # Binary hallucination indicator
    merged_df['is_hallucinated'] = (merged_df['judge_label'] == 2).astype(int)
    
    # Severity score (0 to 1 scale)
    # 0 = correct, 0.33 = partial, 0.67 = hallucinated, 1.0 = we're treating uncertain as high severity
    severity_map = {0: 0.0, 1: 0.33, 2: 0.67, 3: 1.0}
    merged_df['hallucination_severity'] = merged_df['judge_label'].map(severity_map)
    
    # Select and order columns
    columns_to_keep = [
        'id',
        'category',
        'question',
        'ground_truth',
        'model_answer',
        'generation_model',
        'judge_label',
        'judge_confidence',
        'judge_justification',
        'judge_model',
        'is_hallucinated',
        'hallucination_severity',
        'local_id',
        'curvature_score',
        'oppositeness_score',
        'metadata'
    ]
    
    # Keep only columns that exist
    columns_to_keep = [c for c in columns_to_keep if c in merged_df.columns]
    merged_df = merged_df[columns_to_keep]
    
    # Save merged results
    config.results_dir.mkdir(parents=True, exist_ok=True)
    output_file = config.results_dir / "all_results.csv"
    merged_df.to_csv(output_file, index=False)
    logger.info(f"Saved merged results to {output_file}")
    
    # Generate summary statistics
    logger.info("\n=== SUMMARY STATISTICS ===")
    logger.info(f"Total samples: {len(merged_df)}")
    logger.info(f"Hallucinated samples: {merged_df['is_hallucinated'].sum()} ({merged_df['is_hallucinated'].mean()*100:.1f}%)")
    
    logger.info("\nBy category:")
    for category in merged_df['category'].unique():
        cat_data = merged_df[merged_df['category'] == category]
        hall_rate = cat_data['is_hallucinated'].mean() * 100
        logger.info(f"  {category}: {len(cat_data)} samples, {hall_rate:.1f}% hallucinated")
    
    logger.info("\nJudge label distribution:")
    for label in sorted(merged_df['judge_label'].unique()):
        count = (merged_df['judge_label'] == label).sum()
        pct = count / len(merged_df) * 100
        label_name = {0: "Correct", 1: "Partial", 2: "Hallucinated", 3: "Uncertain"}[label]
        logger.info(f"  {label} ({label_name}): {count} ({pct:.1f}%)")
    
    logger.info("\nAggregation complete!")


def main():
    parser = argparse.ArgumentParser(description="Aggregate results")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/config_example.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    aggregate_results(config)


if __name__ == "__main__":
    main()
