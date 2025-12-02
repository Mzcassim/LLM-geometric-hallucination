"""Analyze multi-model consistency.

Aggregates judged results from multiple models and computes:
1. Hallucination rates per model
2. Cross-model correlation of hallucination on specific prompts
3. Geometry vs Hallucination correlation across models
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.io import read_jsonl


def analyze_multi_model(input_dir):
    """Analyze judged results."""
    
    input_path = Path(input_dir)
    files = list(input_path.glob("judged_answers_*.jsonl"))
    
    if not files:
        print(f"No judged files found in {input_dir}")
        return
    
    print(f"Found {len(files)} judged files")
    
    all_data = []
    
    for file in files:
        model_name = file.stem.replace("judged_answers_", "")
        data = read_jsonl(file)
        
        for item in data:
            item['model'] = model_name
            all_data.append(item)
    
    df = pd.DataFrame(all_data)
    
    # Convert label to binary hallucination (1 if label=2, else 0)
    df['is_hallucinated'] = (df['judge_label'] == 2).astype(int)
    
    print(f"Loaded {len(df)} total samples")
    
    # 1. Hallucination Rates
    print("\n" + "="*60)
    print("HALLUCINATION RATES BY MODEL")
    print("="*60)
    
    rates = df.groupby('model')['is_hallucinated'].mean().sort_values()
    print(rates)
    
    # Save rates
    output_dir = input_path.parent
    rates.to_csv(output_dir / "hallucination_rates.csv")
    
    # 2. Consistency Matrix (Correlation between models)
    print("\n" + "="*60)
    print("MODEL CONSISTENCY (CORRELATION)")
    print("="*60)
    
    # Pivot table: rows=prompts, cols=models, values=is_hallucinated
    pivot = df.pivot_table(
        index='id', 
        columns='model', 
        values='is_hallucinated',
        aggfunc='first'  # Should be unique per prompt/model
    )
    
    # Correlation matrix
    corr = pivot.corr()
    print(corr)
    
    # Save correlation
    corr.to_csv(output_dir / "model_consistency_matrix.csv")
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=0, vmax=1)
    plt.title("Hallucination Consistency Across Models")
    plt.tight_layout()
    plt.savefig(output_dir / "consistency_heatmap.png")
    print(f"Saved heatmap to {output_dir / 'consistency_heatmap.png'}")
    
    # 3. Hard Prompts (Hallucinated by >50% of models)
    pivot['failure_rate'] = pivot.mean(axis=1)
    hard_prompts = pivot[pivot['failure_rate'] > 0.5].sort_values('failure_rate', ascending=False)
    
    print("\n" + "="*60)
    print(f"HARD PROMPTS (Failed by >50% of models): {len(hard_prompts)}")
    print("="*60)
    
    # Get question text for hard prompts
    # Create map from id to question
    id_to_question = df.set_index('id')['question'].to_dict()
    
    hard_prompts_list = []
    for prompt_id, row in hard_prompts.iterrows():
        hard_prompts_list.append({
            'id': prompt_id,
            'question': id_to_question.get(prompt_id, "Unknown"),
            'failure_rate': row['failure_rate']
        })
        
        if len(hard_prompts_list) <= 5:
            print(f"{row['failure_rate']:.1%} - {id_to_question.get(prompt_id, 'Unknown')}")
            
    # Save hard prompts
    pd.DataFrame(hard_prompts_list).to_csv(output_dir / "hard_prompts.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Analyze multi-model results")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing judged files")
    
    args = parser.parse_args()
    
    analyze_multi_model(args.input_dir)


if __name__ == "__main__":
    main()
