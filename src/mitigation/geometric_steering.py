"""Geometric steering system.

Automatically detects high-risk queries and rephrases them to move to safer
geometric regions, reducing hallucination risk.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.embedding_client import EmbeddingClient
from src.models.generation_client import GenerationClient
from src.models.judge_client import JudgeClient
from src.geometry.density import compute_local_density
from src.geometry.centrality import compute_distance_to_center
from src.geometry.reference_corpus import load_reference_corpus
from src.config import load_config
from src.utils.seed import set_seed


class GeometricSteerer:
    """Steers queries away from high-risk geometric regions."""
    
    def __init__(self, embed_client, gen_client, reference_corpus):
        self.embed_client = embed_client
        self.gen_client = gen_client
        self.ref_embeddings = reference_corpus['embeddings']
        self.ref_mean = reference_corpus['mean']
    
    def compute_risk_score(self, text):
        """Compute geometric risk score (0-1, higher = riskier)."""
        embedding = self.embed_client.embed_texts([text])[0]
        
        density = compute_local_density(
            embedding.reshape(1, -1),
            self.ref_embeddings,
            k=20,
            metric='cosine'
        )[0]
        
        centrality = compute_distance_to_center(
            embedding.reshape(1, -1),
            self.ref_mean,
            metric='cosine'
        )[0]
        
        # Normalize to 0-1 range (lower density = higher risk, higher centrality = higher risk)
        # Using simple heuristics based on observed ranges
        density_risk = 1 / (1 + density)  # Inverse relationship
        centrality_risk = centrality  # Direct relationship
        
        # Combine (equal weighting)
        risk_score = (density_risk + centrality_risk) / 2
        
        return {
            'risk_score': float(risk_score),
            'density': float(density),
            'centrality': float(centrality),
            'embedding': embedding
        }
    
    def rephrase(self, question):
        """Generate alternative phrasings of the question."""
        
        prompt = f"""Rephrase the following question in 3 different ways while preserving its exact meaning. Make the rephrased versions clearer and more direct.

Original question: {question}

Provide 3 rephrased versions, one per line, numbered 1-3."""
        
        try:
            response = self.gen_client.generate(prompt, max_tokens=300)
            
            # Parse numbered lines
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            rephrasings = []
            
            for line in lines:
                # Remove numbering
                if '. ' in line:
                    parts = line.split('. ', 1)
                    if len(parts) == 2 and parts[0].strip().isdigit():
                        rephrasings.append(parts[1].strip())
                    else:
                        rephrasings.append(line.strip())
                elif line.strip():
                    rephrasings.append(line.strip())
            
            return rephrasings[:3]  # Return at most 3
            
        except Exception as e:
            print(f"Rephrasing error: {e}")
            return []
    
    def steer(self, question, max_attempts=3):
        """Find a safer rephrasing of the question."""
        
        # Compute original risk
        orig_risk = self.compute_risk_score(question)
        
        print(f"\nOriginal question: {question}")
        print(f"  Risk score: {orig_risk['risk_score']:.4f} (density={orig_risk['density']:.4f}, centrality={orig_risk['centrality']:.4f})")
        
        if orig_risk['risk_score'] < 0.5:
            print("  ✓ Already in safe region, no steering needed")
            return {
                'original': question,
                'steered': question,
                'original_risk': orig_risk['risk_score'],
                'final_risk': orig_risk['risk_score'],
                'attempts': 0,
                'success': True
            }
        
        print(f"  ⚠ High risk query! Attempting to steer...")
        
        best_rephrasing = question
        best_risk = orig_risk['risk_score']
        
        for attempt in range(max_attempts):
            print(f"\n  Attempt {attempt + 1}/{max_attempts}:")
            
            # Generate rephrasings
            rephrasings = self.rephrase(question if attempt == 0 else best_rephrasing)
            
            if not rephrasings:
                print("    No rephrasings generated")
                continue
            
            # Evaluate each rephrasing
            for i, rephrase in enumerate(rephrasings):
                risk = self.compute_risk_score(rephrase)
                print(f"    {i+1}. {rephrase}")
                print(f"       Risk: {risk['risk_score']:.4f}")
                
                if risk['risk_score'] < best_risk:
                    best_risk = risk['risk_score']
                    best_rephrasing = rephrase
                    print(f"       ✓ New best!")
            
            # Check if we've achieved safe region
            if best_risk < 0.5:
                print(f"\n  ✓ Achieved safe region!")
                break
        
        risk_reduction = orig_risk['risk_score'] - best_risk
        success = risk_reduction > 0.1  # At least 10% reduction
        
        print(f"\n  Final: Risk reduced by {risk_reduction:+.4f}")
        print(f"  Status: {'✓ SUCCESS' if success else '✗ FAILED'}")
        
        return {
            'original': question,
            'steered': best_rephrasing,
            'original_risk': float(orig_risk['risk_score']),
            'final_risk': float(best_risk),
            'risk_reduction': float(risk_reduction),
            'attempts': attempt + 1 if attempt < max_attempts else max_attempts,
            'success': success
        }


def run_steering_experiment(config, n_samples=10, seed=42):
    """Run geometric steering experiment on high-risk queries."""
    
    set_seed(seed)
    
    # Load V2 results - focus on nonexistent (high-risk category)
    df = pd.read_csv("results/all_results.csv")
    df_nonexistent = df[df['category'] == 'nonexistent']
    
    # Sample high-risk queries (those that hallucinated)
    df_hall = df_nonexistent[df_nonexistent['is_hallucinated'] == 1]
    
    if len(df_hall) < n_samples:
        n_samples = len(df_hall)
    
    sample_indices = np.random.choice(len(df_hall), n_samples, replace=False)
    df_sample = df_hall.iloc[sample_indices]
    
    print(f"Selected {len(df_sample)} high-risk queries from nonexistent category")
    
    # Initialize clients
    embed_client = EmbeddingClient(
        model_name=config.embedding_model,
        batch_size=config.embedding_batch_size
    )
    
    gen_client = GenerationClient(
        model_name=config.generation_model,
        max_retries=config.max_retries,
        timeout=config.api_timeout
    )
    
    judge_client = JudgeClient(
        model_name=config.judge_model,
        max_retries=config.max_retries,
        timeout=config.api_timeout
    )
    
    # Load reference corpus
    ref_corpus = load_reference_corpus(Path("data/reference_corpus"))
    
    # Create steerer
    steerer = GeometricSteerer(embed_client, gen_client, ref_corpus)
    
    # Run steering
    results = []
    
    for idx, row in df_sample.iterrows():
        question = row['question']
        
        print("\n" + "="*60)
        print(f"Query {len(results) + 1}/{len(df_sample)}")
        
        # Steer
        steering_result = steerer.steer(question, max_attempts=2)
        
        # Generate answer with steered question
        steered_q = steering_result['steered']
        
        if steered_q != question:
            print(f"\nGenerating answer for steered question...")
            try:
                steered_answer = gen_client.generate(steered_q, max_tokens=200)
                
                # Judge steered answer
                steered_judgment = judge_client.judge(
                    question=steered_q,
                    answer=steered_answer,
                    ground_truth=row.get('ground_truth', 'N/A')
                )
                
                steered_hall = steered_judgment['label'] == 2
                
                print(f"  Steered result: {'HALLUCINATED' if steered_hall else 'Correct'}")
                
                steering_result['steered_answer'] = steered_answer
                steering_result['steered_judge_label'] = int(steered_judgment['label'])
                steering_result['steered_is_hallucinated'] = int(steered_hall)
                steering_result['steered_confidence'] = float(steered_judgment['confidence'])
                
            except Exception as e:
                print(f"  Error generating/judging steered version: {e}")
        
        # Store original info
        steering_result['original_answer'] = row.get('model_answer', 'N/A')
        steering_result['original_is_hallucinated'] = int(row['is_hallucinated'])
        
        results.append(steering_result)
    
    return results


def analyze_steering_results(results):
    """Analyze steering effectiveness."""
    
    print("\n" + "="*60)
    print("STEERING RESULTS SUMMARY")
    print("="*60)
    
    df = pd.DataFrame(results)
    
    # Risk reduction
    mean_risk_reduction = df['risk_reduction'].mean()
    print(f"\nAverage risk reduction: {mean_risk_reduction:+.4f}")
    print(f"Successful steering: {df['success'].sum()}/{len(df)} ({df['success'].mean():.1%})")
    
    # Hallucination reduction (if we have steered judgments)
    if 'steered_is_hallucinated' in df.columns:
        orig_hall_rate = df['original_is_hallucinated'].mean()
        steered_hall_rate = df['steered_is_hallucinated'].mean()
        reduction = orig_hall_rate - steered_hall_rate
        
        print(f"\nHallucination rates:")
        print(f"  Original: {orig_hall_rate:.1%}")
        print(f"  Steered: {steered_hall_rate:.1%}")
        print(f"  Reduction: {reduction:+.1%}")
    
    return df


def main():
    """Run geometric steering experiment."""
    
    print("="*60)
    print("GEOMETRIC STEERING SYSTEM")
    print("="*60)
    
    config = load_config("experiments/config_v2.yaml")
    
    # Run experiment
    results = run_steering_experiment(config, n_samples=5, seed=42)
    
    # Analyze
    df_results = analyze_steering_results(results)
    
    # Save
    output_dir = Path("results/v3")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "geometric_steering.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    csv_file = output_dir / "geometric_steering.csv"
    df_results.to_csv(csv_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print("="*60)


if __name__ == "__main__":
    main()
