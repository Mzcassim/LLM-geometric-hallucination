"""Parallel execution manager for multi-model generation.

Runs models in parallel while respecting provider-specific rate limits.
Groups models by provider and runs one from each provider concurrently.
"""

import sys
import time
import subprocess
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def load_config(config_path="experiments/multi_model_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_model(model_key, n_prompts, output_dir):
    """Run generation for a single model as a subprocess."""
    print(f"🚀 Starting {model_key}...")
    cmd = [
        "python3", "-m", "src.pipeline.run_multi_model_generation",
        "--model-key", model_key,
        "--n-prompts", str(n_prompts),
        "--output-dir", output_dir
    ]
    
    # Run and stream output
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
        for line in p.stdout:
            print(f"[{model_key}] {line.strip()}")
            
    if p.returncode == 0:
        print(f"✅ Finished {model_key}")
        return True
    else:
        print(f"❌ Failed {model_key}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-prompts", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results/v3/multi_model")
    args = parser.parse_args()
    
    config = load_config()
    models = config['models']
    
    # Group models by provider
    providers = {}
    for m in models:
        p = m['provider']
        if p not in providers:
            providers[p] = []
        providers[p].append(m['name'])
        
    print(f"Found {len(models)} models across {len(providers)} providers: {list(providers.keys())}")
    
    # Strategy: Run one model from each provider in parallel
    # This maximizes throughput without hitting single-provider rate limits
    
    max_workers = len(providers)  # One worker per provider
    print(f"Running with {max_workers} parallel workers (one per provider)...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        # For each provider, submit their models sequentially
        for provider, model_list in providers.items():
            def run_provider_models(models_to_run):
                for m in models_to_run:
                    success = run_model(m, args.n_prompts, args.output_dir)
                    if not success:
                        print(f"⚠️ Skipping remaining models for {provider} due to failure")
                        # Optional: break or continue depending on desired robustness
                        # continue 
                return True
            
            futures.append(executor.submit(run_provider_models, model_list))
            
        # Wait for all to complete
        for f in futures:
            f.result()

if __name__ == "__main__":
    main()
