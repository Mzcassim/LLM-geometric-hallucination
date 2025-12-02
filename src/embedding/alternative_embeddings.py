"""Compute alternative embeddings for robustness analysis.

This script generates embeddings using different models to test if 
geometry-hallucination correlations are robust to embedding choice.

Models:
- text-embedding-3-large (OpenAI, 3072-dim)
- all-mpnet-base-v2 (Open source, 768-dim)
"""

import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.io import read_jsonl


def embed_with_openai_large(texts, output_file):
    """Embed using text-embedding-3-large."""
    from openai import OpenAI
    
    client = OpenAI()
    embeddings = []
    
    print(f"Embedding {len(texts)} texts with text-embedding-3-large...")
    for text in tqdm(texts):
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        embeddings.append(response.data[0].embedding)
    
    embeddings_array = np.array(embeddings)
    np.save(output_file, embeddings_array)
    print(f"Saved to {output_file} (shape: {embeddings_array.shape})")
    
    return embeddings_array


def embed_with_sentence_transformers(texts, output_file, model_name='all-mpnet-base-v2'):
    """Embed using Sentence Transformers (open source)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Error: sentence-transformers not installed")
        print("Install with: pip install sentence-transformers")
        return None
    
    print(f"Loading Sentence Transformer model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    print(f"Embedding {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    
    np.save(output_file, embeddings)
    print(f"Saved to {output_file} (shape: {embeddings.shape})")
    
    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-file", default="data/prompts/prompts.jsonl")
    parser.add_argument("--output-dir", default="data/processed/alternative_embeddings")
    parser.add_argument("--models", nargs='+', default=['openai-large', 'mpnet'], 
                       choices=['openai-large', 'mpnet'])
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    prompts = read_jsonl(args.prompts_file)
    texts = [p['question'] for p in prompts]
    
    print(f"Loaded {len(texts)} prompts")
    print("")
    
    # Generate embeddings for each model
    for model_type in args.models:
        if model_type == 'openai-large':
            output_file = Path(args.output_dir) / "embeddings_openai_large.npy"
            embed_with_openai_large(texts, output_file)
            
        elif model_type == 'mpnet':
            output_file = Path(args.output_dir) / "embeddings_mpnet.npy"
            embed_with_sentence_transformers(texts, output_file, 'all-mpnet-base-v2')
        
        print("")
    
    print("Done! Alternative embeddings saved.")


if __name__ == "__main__":
    main()
