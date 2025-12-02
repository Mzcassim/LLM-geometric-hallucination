"""Human verification tool for judging quality.

Samples a subset of judgments from ALL experimental conditions:
- Base multi-model (Option 2)
- Adversarial attacks (Option 3)
- Geometric steering (Option 3)

Calculates agreement rate between AI judges and human ground truth.
"""

import sys
import json
import random
import pandas as pd
from pathlib import Path
from src.utils.io import read_jsonl

def verify_judgments(input_dirs, n_samples=50):
    """Interactive verification of judgments from multiple sources."""
    
    # Load all judgments from all directories
    all_records = []
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        
        if not input_path.exists():
            print(f"Directory not found: {input_dir} (skipping)")
            continue
        
        files = list(input_path.glob("judged_*.jsonl")) + list(input_path.glob("judged_answers_*.jsonl"))
        
        for f in files:
            records = read_jsonl(f)
            for r in records:
                r['source_file'] = f.name
                r['source_dir'] = input_dir
                all_records.append(r)
    
    if not all_records:
        print("No records found in any directory.")
        return
        
    print(f"Loaded {len(all_records)} judgments from {len(input_dirs)} sources")
    
    # Group by source
    by_source = {}
    for r in all_records:
        source = r['source_dir']
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(r)
    
    print(f"\nDistribution:")
    for source, records in by_source.items():
        print(f"  {source}: {len(records)} judgments")
    
    # Sample proportionally from each source
    sample = []
    for source, records in by_source.items():
        # Calculate proportion
        proportion = len(records) / len(all_records)
        n_from_source = max(1, int(n_samples * proportion))  # At least 1 from each
        
        # Within source, stratify by hallucination status
        hallucinations = [r for r in records if r.get('judge_label') == 2]
        others = [r for r in records if r.get('judge_label') != 2]
        
        n_hall = min(n_from_source // 2, len(hallucinations))
        n_other = n_from_source - n_hall
        
        if len(hallucinations) >= n_hall and len(others) >= n_other:
            sample.extend(random.sample(hallucinations, n_hall))
            sample.extend(random.sample(others, n_other))
        else:
            # Fallback: just sample what we can
            sample.extend(random.sample(records, min(n_from_source, len(records))))
    
    # Ensure we have exactly n_samples (adjust if needed)
    if len(sample) > n_samples:
        sample = random.sample(sample, n_samples)
    elif len(sample) < n_samples:
        # Add more if we're short
        remaining = [r for r in all_records if r not in sample]
        if remaining:
            sample.extend(random.sample(remaining, min(n_samples - len(sample), len(remaining))))
    
    random.shuffle(sample)
    
    print(f"\nStarting verification of {len(sample)} samples...")
    print("Instructions: Read the Question, Answer, and Evidence.")
    print("Rate the answer: 0=Correct, 1=Partial, 2=Hallucinated, 3=Uncertain")
    print("Press 'q' to quit early.\n")
    
    human_labels = []
    ai_labels = []
    sources = []
    
    for i, record in enumerate(sample):
        print(f"\n--- Sample {i+1}/{len(sample)} ---")
        print(f"Source: {record['source_dir']}")
        print(f"Model: {record.get('model', record.get('model_name', 'Unknown'))}")
        print(f"Question: {record['question']}")
        print(f"Answer:   {record.get('model_answer', record.get('answer', ''))}")
        print(f"Evidence: {record.get('ground_truth', record.get('evidence', ''))}")
        print(f"AI Verdict: {record['judge_label']} (Conf: {record.get('judge_confidence', 0):.2f})")
        print(f"AI Reason:  {record.get('judge_justification', '')[:150]}...")
        
        while True:
            user_input = input("Your rating [0/1/2/3]: ").strip().lower()
            if user_input == 'q':
                print("\nQuitting early...")
                break
            if user_input in ['0', '1', '2', '3']:
                human_labels.append(int(user_input))
                ai_labels.append(record['judge_label'])
                sources.append(record['source_dir'])
                break
            print("Invalid input. Please enter 0, 1, 2, or 3.")
        
        if user_input == 'q':
            break
    
    if not human_labels:
        print("No labels collected.")
        return
    
    # Calculate agreement
    matches = sum(1 for h, a in zip(human_labels, ai_labels) if h == a)
    accuracy = matches / len(human_labels)
    
    print("\n--- Verification Results ---")
    print(f"Samples Verified: {len(human_labels)}")
    print(f"Agreement Rate:   {accuracy:.1%}")
    
    # Agreement by source
    print("\nAgreement by Source:")
    for source in set(sources):
        source_matches = sum(1 for h, a, s in zip(human_labels, ai_labels, sources) 
                            if h == a and s == source)
        source_total = sum(1 for s in sources if s == source)
        if source_total > 0:
            print(f"  {source}: {source_matches}/{source_total} ({source_matches/source_total:.1%})")
    
    # Save report
    report = {
        "samples": len(human_labels),
        "accuracy": accuracy,
        "human_labels": human_labels,
        "ai_labels": ai_labels,
        "sources": sources
    }
    
    output_file = "results/v3/human_verification_report.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", nargs='+', 
                       default=["results/v3/multi_model/judged", 
                               "results/v3/attacks/judged",
                               "results/v3/steering/judged"],
                       help="Directories containing judged results")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    
    verify_judgments(args.input_dirs, args.n)
