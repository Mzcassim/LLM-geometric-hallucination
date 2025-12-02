"""Run judging for multi-model experiment.

Iterates through all answer files in the input directory and judges them.
"""

import sys
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.judge_client import JudgeClient
from src.utils.io import read_jsonl, write_jsonl
from src.config import load_config


def run_multi_model_judging(input_dir, output_dir, config_path="experiments/config_v2.yaml"):
    """Run judging for all answer files in input_dir."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load config for judge model settings
    config = load_config(config_path)
    
    # Initialize judge client
    print(f"Initializing judge client ({config.judge_model})...")
    judge_client = JudgeClient(
        model_name=config.judge_model,
        max_retries=config.max_retries,
        timeout=config.api_timeout
    )
    
    # Find answer files
    answer_files = list(input_path.glob("answers_*.jsonl"))
    print(f"Found {len(answer_files)} answer files to judge")
    
    for file in answer_files:
        model_name = file.stem.replace("answers_", "")
        print(f"\nJudging answers for {model_name}...")
        
        answers = read_jsonl(file)
        results = []
        
        output_file = output_path / f"judged_{file.name}"
        if output_file.exists():
            print(f"  Skipping (already exists): {output_file}")
            continue
            
        for answer_data in tqdm(answers):
            try:
                judgment = judge_client.judge(
                    question=answer_data["question"],
                    answer=answer_data["model_answer"],
                    ground_truth=answer_data["ground_truth"],
                    meta_info=answer_data.get("metadata", {})
                )
                
                result = {
                    **answer_data,
                    "judge_label": judgment["label"],
                    "judge_confidence": judgment["confidence"],
                    "judge_justification": judgment["justification"],
                    "judge_model": config.judge_model
                }
                results.append(result)
                
            except Exception as e:
                print(f"  Error judging {answer_data.get('id', 'unknown')}: {e}")
                # Add failed entry
                result = {
                    **answer_data,
                    "judge_label": 3,  # Uncertain
                    "judge_confidence": 0.0,
                    "judge_justification": f"Error: {str(e)}",
                    "judge_model": config.judge_model
                }
                results.append(result)
        
        write_jsonl(output_file, results)
        print(f"  Saved {len(results)} judgments to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run multi-model judging")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing answers_*.jsonl files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save judged files")
    
    args = parser.parse_args()
    
    run_multi_model_judging(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
