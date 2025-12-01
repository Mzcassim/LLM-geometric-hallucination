"""Run judging phase: evaluate model answers using LLM-as-a-judge."""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import load_config, ProjectConfig
from src.models.judge_client import JudgeClient
from src.utils.io import read_jsonl, write_jsonl
from src.utils.logging_utils import setup_logger, log_progress
from src.utils.seed import set_seed


def run_judging(config: ProjectConfig):
    """Run the judging phase."""
    logger = setup_logger(
        "judging",
        log_file=config.data_dir / "logs" / "judging.log"
    )
    
    # Set random seed
    set_seed(config.seed)
    
    logger.info(f"Starting judging with model: {config.judge_model}")
    
    # Initialize judge client
    judge_client = JudgeClient(
        model_name=config.judge_model,
        max_retries=config.max_retries,
        timeout=config.api_timeout
    )
    
    # Load model answers
    answers_file = config.processed_dir / "model_answers.jsonl"
    if not answers_file.exists():
        logger.error(f"Model answers file not found: {answers_file}")
        logger.error("Please run run_generation.py first")
        return
    
    answers = read_jsonl(answers_file)
    logger.info(f"Loaded {len(answers)} model answers")
    
    # Judge each answer
    results = []
    output_file = config.processed_dir / "judged_answers.jsonl"
    
    for i, answer_data in enumerate(answers):
        if i % 10 == 0:
            log_progress(logger, i, len(answers), "Judging ")
        
        try:
            # Get judgment
            judgment = judge_client.judge(
                question=answer_data["question"],
                answer=answer_data["model_answer"],
                ground_truth=answer_data["ground_truth"],
                meta_info=answer_data.get("metadata", {})
            )
            
            # Create result entry
            result = {
                **answer_data,  # Include all original fields
                "judge_label": judgment["label"],
                "judge_confidence": judgment["confidence"],
                "judge_justification": judgment["justification"],
                "judge_model": config.judge_model
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to judge answer for {answer_data['id']}: {e}")
            # Add failed entry with default values
            result = {
                **answer_data,
                "judge_label": 3,  # Uncertain
                "judge_confidence": 0.0,
                "judge_justification": f"Error during judging: {str(e)}",
                "judge_model": config.judge_model
            }
            results.append(result)
    
    # Save results
    write_jsonl(output_file, results)
    logger.info(f"Judging complete! Saved {len(results)} judgments to {output_file}")
    
    # Log summary statistics
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for r in results:
        label_counts[r["judge_label"]] += 1
    
    logger.info("Label distribution:")
    logger.info(f"  Correct (0): {label_counts[0]}")
    logger.info(f"  Partial (1): {label_counts[1]}")
    logger.info(f"  Hallucinated (2): {label_counts[2]}")
    logger.info(f"  Uncertain/Refusal (3): {label_counts[3]}")


def main():
    parser = argparse.ArgumentParser(description="Run judging phase")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/config_example.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_judging(config)


if __name__ == "__main__":
    main()
