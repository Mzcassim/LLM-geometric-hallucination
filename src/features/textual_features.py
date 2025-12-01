"""Extract textual features from questions.

Computes simple text statistics to use as controls for ruling out
that geometry is just re-encoding lexical properties.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def extract_textual_features(text):
    """Extract lexical and syntactic features from text."""
    
    # Basic counts
    tokens = text.lower().split()
    n_tokens = len(tokens)
    n_chars = len(text)
    n_unique_tokens = len(set(tokens))
    
    # Lexical diversity
    lexical_diversity = n_unique_tokens / n_tokens if n_tokens > 0 else 0
    
    # Average word length
    avg_word_len = np.mean([len(w) for w in tokens]) if tokens else 0
    
    # Question type indicators
    question_words = ['who', 'what', 'when', 'where', 'why', 'how', 'which', 'whose']
    has_question_word = any(qw in text.lower() for qw in question_words)
    question_word = next((qw for qw in question_words if qw in text.lower()), None)
    
    # Rare/unusual patterns
    # Character trigram weirdness (rough proxy for "weird names")
    char_trigrams = [text[i:i+3] for i in range(len(text) - 2)]
    trigram_counts = Counter(char_trigrams)
    trigram_entropy = -sum((count/len(char_trigrams)) * np.log2(count/len(char_trigrams)) 
                           for count in trigram_counts.values()) if char_trigrams else 0
    
    # Capital letters (proxy for proper nouns)
    n_capitals = sum(1 for ch in text if ch.isupper())
    capital_ratio = n_capitals / n_chars if n_chars > 0 else 0
    
    # Punctuation
    n_punctuation = sum(1 for ch in text if ch in '.,!?;:()"\'')
    punct_ratio = n_punctuation / n_chars if n_chars > 0 else 0
    
    # Numbers
    has_numbers = bool(re.search(r'\d', text))
    n_numbers = len(re.findall(r'\d+', text))
    
    # Discourse markers (uncertainty, hedging)
    uncertainty_markers = ['might', 'may', 'perhaps', 'possibly', 'allegedly', 
                          'reportedly', 'supposedly', 'apparently']
    n_uncertainty = sum(1 for marker in uncertainty_markers if marker in text.lower())
    
    # Specific keywords that might correlate with categories
    academic_words = ['theorem', 'prove', 'discovery', 'theory', 'formula', 
                     'principle', 'law', 'equation']
    n_academic = sum(1 for word in academic_words if word in text.lower())
    
    fiction_words = ['wrote', 'book', 'novel', 'story', 'character', 'author']
    n_fiction = sum(1 for word in fiction_words if word in text.lower())
    
    return {
        'n_tokens': n_tokens,
        'n_chars': n_chars,
        'n_unique_tokens': n_unique_tokens,
        'lexical_diversity': lexical_diversity,
        'avg_word_len': avg_word_len,
        'has_question_word': int(has_question_word),
        'question_word': question_word if question_word else 'none',
        'trigram_entropy': trigram_entropy,
        'n_capitals': n_capitals,
        'capital_ratio': capital_ratio,
        'n_punct': n_punctuation,
        'punct_ratio': punct_ratio,
        'has_numbers': int(has_numbers),
        'n_numbers': n_numbers,
        'n_uncertainty_markers': n_uncertainty,
        'n_academic_words': n_academic,
        'n_fiction_words': n_fiction
    }


def compute_textual_features_for_dataset(df):
    """Compute textual features for all questions in dataset."""
    
    print("Computing textual features...")
    
    features_list = []
    
    for idx, row in df.iterrows():
        question = row['question']
        features = extract_textual_features(question)
        features['id'] = row['id']
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    
    # Set id as index and join
    features_df = features_df.set_index('id')
    
    print(f"Extracted {len(features_df.columns)} textual features")
    print(f"Features: {list(features_df.columns)}")
    
    return features_df


def main():
    """Extract textual features from V2 results."""
    
    # Load results
    results_file = Path("results/all_results.csv")
    if not results_file.exists():
        print(f"Error: {results_file} not found")
        return
    
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} questions")
    
    # Compute features
    features_df = compute_textual_features_for_dataset(df)
    
    # Save
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "textual_features.csv"
    features_df.to_csv(output_file)
    
    print(f"\n✓ Saved to {output_file}")
    
    # Show summary statistics
    print("\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)
    
    print("\nNumerical features:")
    print(features_df.select_dtypes(include=[np.number]).describe())
    
    print("\nCategorical features:")
    print(features_df.select_dtypes(include=['object']).describe())
    
    print("\nQuestion word distribution:")
    print(features_df['question_word'].value_counts())


if __name__ == "__main__":
    main()
