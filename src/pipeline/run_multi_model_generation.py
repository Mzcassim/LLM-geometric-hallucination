"""Run generation pipeline for multiple models.

Iterates through configured models, generates responses for benchmark prompts,
and saves them for judging.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import json
import time
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.multi_model_client import get_model_client
from src.utils.io import read_jsonl, write_jsonl
from src.utils.seed import set_seed


def load_prompts(data_dir, n_prompts=None):
    """Load prompts from all categories."""
    prompts = []
    prompt_files = list(Path(data_dir).glob("*.jsonl"))
    
    for file in prompt_files:
        category_prompts = read_jsonl(file)
        if n_prompts and n_prompts > 0:
            # Deterministic sampling
            import random
            random.seed(42)
            if len(category_prompts) > n_prompts:
                category_prompts = random.sample(category_prompts, n_prompts)
        
        prompts.extend(category_prompts)
    
    return prompts


def run_generation(model_key, n_prompts, output_dir):
    """Run generation for a specific model with resume capability."""
    
    print(f"Initializing client for {model_key}...")
    try:
        client = get_model_client(model_key)
    except Exception as e:
        print(f"Error initializing client: {e}")
        return
    
    # Load prompts
    prompts = load_prompts("data/prompts", n_prompts)
    print(f"Loaded {len(prompts)} prompts")
    
    # Check for existing results (resume capability)
    output_path = Path(output_dir) / f"answers_{model_key}.jsonl"
    existing_ids = set()
    
    if output_path.exists():
        print(f"Found existing results, loading for resume...")
        try:
            existing_results = read_jsonl(output_path)
            existing_ids = {r['id'] for r in existing_results}
            print(f"  Resuming: {len(existing_ids)} already completed, {len(prompts) - len(existing_ids)} remaining")
        except Exception as e:
            print(f"  Warning: Could not load existing results: {e}")
            print(f"  Starting fresh...")
    
    results = []
    skipped = 0
    failed = 0
    
    print(f"Generating responses with {model_key}...")
    for prompt_data in tqdm(prompts):
        # Skip if already processed
        if prompt_data['id'] in existing_ids:
            skipped += 1
            continue
            
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                response = client.generate(
                    prompt=prompt_data['question'],
                    max_tokens=4000,
                    temperature=0.7
                )
                
                result = prompt_data.copy()
                result['model'] = model_key
                result['model_answer'] = response
                results.append(result)
                break  # Success, exit retry loop
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a timeout or rate limit
                if "timeout" in error_msg.lower():
                    print(f"\n  Timeout on prompt {prompt_data['id']}, attempt {attempt+1}/{max_retries}")
                elif "rate_limit" in error_msg.lower() or "429" in error_msg:
                    print(f"\n  Rate limit hit, waiting {retry_delay}s before retry...")
                else:
                    print(f"\n  Error on prompt {prompt_data['id']}: {error_msg}")
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed
                    print(f"\n  Failed after {max_retries} attempts: {prompt_data['id']}")
                    failed += 1
    
    # Append new results to file (preserving existing ones)
    if results:
        if existing_ids:
            # Append mode
            with open(output_path, 'a', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"\nAppended {len(results)} new responses to {output_path}")
        else:
            # Fresh write
            write_jsonl(output_path, results)
            print(f"\nSaved {len(results)} responses to {output_path}")
    
    # Summary
    total_expected = len(prompts)
    total_completed = len(existing_ids) + len(results)
    print(f"\nSummary for {model_key}:")
    print(f"  Total prompts: {total_expected}")
    print(f"  Completed: {total_completed}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {(total_completed / total_expected * 100):.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Run multi-model generation")
    parser.add_argument("--model-key", type=str, required=True, help="Model key from config")
    parser.add_argument("--n-prompts", type=int, default=50, help="Number of prompts per category")
    parser.add_argument("--output-dir", type=str, default="results/v3/multi_model", help="Output directory")
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    run_generation(args.model_key, args.n_prompts, args.output_dir)


if __name__ == "__main__":
    main()
