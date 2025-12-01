"""Run generation phase: get model answers for all benchmark questions."""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import load_config, ProjectConfig
from src.models.generation_client import GenerationClient
from src.utils.io import read_jsonl, write_jsonl
from src.utils.logging_utils import setup_logger, log_progress
from src.utils.seed import set_seed


def run_generation(config: ProjectConfig):
    """Run the generation phase."""
    logger = setup_logger(
        "generation",
        log_file=config.data_dir / "logs" / "generation.log"
    )
    
    # Set random seed
    set_seed(config.seed)
    
    logger.info(f"Starting generation with model: {config.generation_model}")
    
    # Initialize generation client
    gen_client = GenerationClient(
        model_name=config.generation_model,
        max_retries=config.max_retries,
        timeout=config.api_timeout
    )
    
    # Load all prompts
    all_prompts = []
    prompts_dir = config.prompts_dir
    
    for prompt_file in prompts_dir.glob("*.jsonl"):
        logger.info(f"Loading prompts from {prompt_file.name}")
        prompts = read_jsonl(prompt_file)
        all_prompts.extend(prompts)
    
    logger.info(f"Loaded {len(all_prompts)} total prompts")
    
    # Limit prompts if specified
    if config.max_prompts_per_category < float('inf'):
        # Group by category and limit each
        from collections import defaultdict
        by_category = defaultdict(list)
        for p in all_prompts:
            by_category[p['category']].append(p)
        
        limited_prompts = []
        for category, prompts in by_category.items():
            limited = prompts[:config.max_prompts_per_category]
            limited_prompts.extend(limited)
            logger.info(f"Category '{category}': using {len(limited)}/{len(prompts)} prompts")
        
        all_prompts = limited_prompts
    
    # Generate answers
    results = []
    output_file = config.processed_dir / "model_answers.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    for i, prompt_data in enumerate(all_prompts):
        if i % 10 == 0:
            log_progress(logger, i, len(all_prompts), "Generation ")
        
        try:
            # Generate answer
            answer = gen_client.generate(prompt_data["question"])
            
            # Create result entry
            result = {
                "id": prompt_data["id"],
                "category": prompt_data["category"],
                "question": prompt_data["question"],
                "ground_truth": prompt_data["ground_truth"],
                "model_answer": answer,
                "generation_model": config.generation_model,
                "metadata": prompt_data.get("metadata", {})
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to generate answer for {prompt_data['id']}: {e}")
            # Add failed entry
            result = {
                "id": prompt_data["id"],
                "category": prompt_data["category"],
                "question": prompt_data["question"],
                "ground_truth": prompt_data["ground_truth"],
                "model_answer": f"[ERROR: {str(e)}]",
                "generation_model": config.generation_model,
                "metadata": prompt_data.get("metadata", {})
            }
            results.append(result)
    
    # Save results
    write_jsonl(output_file, results)
    logger.info(f"Generation complete! Saved {len(results)} answers to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run generation phase")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/config_example.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_generation(config)


if __name__ == "__main__":
    main()
