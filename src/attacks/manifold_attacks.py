"""Adversarial manifold attacks.

Tests whether intentionally perturbing prompts to push them into low-density
regions increases hallucination rate (causal proof of geometry → hallucination).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import random

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.embedding_client import EmbeddingClient
from src.models.generation_client import GenerationClient
from src.models.judge_client import JudgeClient
from src.geometry.density import compute_local_density
from src.geometry.centrality import compute_distance_to_center
from src.geometry.reference_corpus import load_reference_corpus
from src.config import load_config
from src.utils.seed import set_seed


class ManifoldAttacker:
    """Performs adversarial perturbations to manipulate embedding geometry."""
    
    def __init__(self, embed_client, reference_corpus):
        self.embed_client = embed_client
        self.ref_embeddings = reference_corpus['embeddings']
        self.ref_mean = reference_corpus['mean']
    
    def compute_geometry(self, text):
        """Compute geometric features for a text."""
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
        
        return {
            'density': float(density),
            'centrality': float(centrality),
            'embedding': embedding
        }
    
    def add_confusing_context(self, question):
        """Add confusing/contradictory context to push into uncertain region."""
        perturbations = [
            f"{question} (Note: Some historians dispute this.)",
            f"{question} Although there are conflicting accounts.",
            f"{question} Despite ongoing debates about this topic.",
            f"According to some sources: {question}",
            f"{question} However, reliable sources differ on this."
        ]
        return random.choice(perturbations)
    
    def replace_with_synonyms(self, question):
        """Replace common words with rare synonyms."""
        synonym_map = {
            'who': 'which individual',
            'what': 'which entity',
            'where': 'at what location',
            'when': 'at what time',
            'how': 'in what manner',
            'many': 'numerous',
            'large': 'substantial',
            'small': 'diminutive',
            'first': 'inaugural',
            'last': 'terminal'
        }
        
        perturbed = question
        for word, syn in synonym_map.items():
            if word in question.lower():
                perturbed = perturbed.replace(word, syn).replace(word.title(), syn.title())
                break
        
        return perturbed
    
    def add_noise_tokens(self, question):
        """Add semantically-disconnected tokens."""
        noise_phrases = [
            "regarding computational linguistics",
            "within the context of modern historiography",
            "from a contemporary perspective",
            "considering recent scholarship",
            "in light of current understanding"
        ]
        
        # Insert noise at random position
        words = question.split()
        if len(words) > 2:
            pos = random.randint(1, len(words) - 1)
            words.insert(pos, random.choice(noise_phrases))
        
        return ' '.join(words)
    
    def perturb(self, question, method='confusing'):
        """Apply perturbation method."""
        if method == 'confusing':
            return self.add_confusing_context(question)
        elif method == 'synonyms':
            return self.replace_with_synonyms(question)
        elif method == 'noise':
            return self.add_noise_tokens(question)
        else:
            return question


def run_attack_experiment(config, n_samples=20, seed=42):
    """Run adversarial attack experiment on factual questions."""
    
    set_seed(seed)
    random.seed(seed)
    
    # Load V2 results (factual only, correct answers)
    df = pd.read_csv("results/all_results.csv")
    df_factual = df[(df['category'] == 'factual') & (df['is_hallucinated'] == 0)]
    
    if len(df_factual) < n_samples:
        print(f"Only {len(df_factual)} factual correct samples available")
        n_samples = len(df_factual)
    
    # Sample questions
    sample_indices = np.random.choice(len(df_factual), n_samples, replace=False)
    df_sample = df_factual.iloc[sample_indices].copy()
    
    print(f"Selected {len(df_sample)} factual questions that were answered correctly")
    
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
    
    # Create attacker
    attacker = ManifoldAttacker(embed_client, ref_corpus)
    
    # Run attacks
    results = []
    methods = ['confusing', 'synonyms', 'noise']
    
    for idx, row in df_sample.iterrows():
        original_q = row['question']
        
        print(f"\n{len(results) + 1}. Original: {original_q}")
        
        # Compute original geometry
        orig_geo = attacker.compute_geometry(original_q)
        print(f"   Original geometry: density={orig_geo['density']:.4f}, centrality={orig_geo['centrality']:.4f}")
        
        # Try each perturbation method
        for method in methods:
            perturbed_q = attacker.perturb(original_q, method=method)
            
            if perturbed_q == original_q:
                continue
            
            print(f"   {method}: {perturbed_q}")
            
            # Compute perturbed geometry
            pert_geo = attacker.compute_geometry(perturbed_q)
            print(f"   Perturbed geometry: density={pert_geo['density']:.4f}, centrality={pert_geo['centrality']:.4f}")
            
            # Generate answer
            try:
                answer = gen_client.generate(perturbed_q, max_tokens=200)
                
                # Judge
                judgment = judge_client.judge(
                    question=perturbed_q,
                    answer=answer,
                    ground_truth=row.get('ground_truth', 'N/A')
                )
                
                is_hall = judgment['label'] == 2  # 2 = hallucinated
                
                print(f"   Result: {'HALLUCINATED' if is_hall else 'Correct'} (label={judgment['label']})")
                
                results.append({
                    'original_question': original_q,
                    'perturbed_question': perturbed_q,
                    'perturbation_method': method,
                    'original_density': orig_geo['density'],
                    'perturbed_density': pert_geo['density'],
                    'original_centrality': orig_geo['centrality'],
                    'perturbed_centrality': pert_geo['centrality'],
                    'density_change': pert_geo['density'] - orig_geo['density'],
                    'centrality_change': pert_geo['centrality'] - orig_geo['centrality'],
                    'answer': answer,
                    'judge_label': int(judgment['label']),
                    'is_hallucinated': int(is_hall),
                    'judge_confidence': float(judgment['confidence']),
                    'judge_justification': judgment['justification']
                })
                
            except Exception as e:
                print(f"   Error: {e}")
    
    return results


def analyze_attack_results(results):
    """Analyze and summarize attack results."""
    
    print("\n" + "="*60)
    print("ATTACK RESULTS SUMMARY")
    print("="*60)
    
    df = pd.DataFrame(results)
    
    # Overall hallucination rate increase
    overall_hall_rate = df['is_hallucinated'].mean()
    print(f"\nOverall hallucination rate after perturbation: {overall_hall_rate:.1%}")
    
    # By perturbation method
    print("\nBy perturbation method:")
    for method in df['perturbation_method'].unique():
        method_df = df[df['perturbation_method'] == method]
        hall_rate = method_df['is_hallucinated'].mean()
        print(f"  {method:12s}: {hall_rate:.1%} ({method_df['is_hallucinated'].sum()}/{len(method_df)})")
    
    # Geometry changes
    print("\nGeometry changes:")
    print(f"  Density change: {df['density_change'].mean():+.4f} ± {df['density_change'].std():.4f}")
    print(f"  Centrality change: {df['centrality_change'].mean():+.4f} ± {df['centrality_change'].std():.4f}")
    
    # Correlation between geometry change and hallucination
    from scipy.stats import pointbiserialr
    
    if df['is_hallucinated'].sum() > 0 and df['is_hallucinated'].sum() < len(df):
        corr_density, p_density = pointbiserialr(df['is_hallucinated'], df['density_change'])
        corr_centrality, p_centrality = pointbiserialr(df['is_hallucinated'], df['centrality_change'])
        
        print("\nCorrelation with hallucination:")
        print(f"  Density change: r={corr_density:+.3f}, p={p_density:.4f}")
        print(f"  Centrality change: r={corr_centrality:+.3f}, p={p_centrality:.4f}")
    
    return df


def main():
    """Run adversarial manifold attack experiment."""
    
    print(" ="*60)
    print("ADVERSARIAL MANIFOLD ATTACKS")
    print("="*60)
    
    config = load_config("experiments/config_v2.yaml")
    
    # Run experiment
    results = run_attack_experiment(config, n_samples=10, seed=42)
    
    # Analyze
    df_results = analyze_attack_results(results)
    
    # Save
    output_dir = Path("results/v3")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "adversarial_attacks.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    csv_file = output_dir / "adversarial_attacks.csv"
    df_results.to_csv(csv_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"CSV saved to: {csv_file}")
    print("="*60)


if __name__ == "__main__":
    main()
