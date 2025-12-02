#!/bin/bash
# Optional: Embedding Robustness Analysis
# Tests if geometry-hallucination correlations hold with alternative embeddings

set -e

echo "================================================================================"
echo "EMBEDDING ROBUSTNESS ANALYSIS (OPTIONAL)"
echo "================================================================================"
echo ""
echo "This workflow tests if results are robust to embedding model choice."
echo "It compares:"
echo "  1. text-embedding-3-small (Original, 1536-dim)"
echo "  2. text-embedding-3-large (OpenAI, 3072-dim)"
echo "  3. all-mpnet-base-v2 (Open Source, 768-dim)"
echo ""
echo "⚠️  Note: This requires additional API calls and the sentence-transformers library."
echo ""

read -p "Continue? [y/n]: " CONFIRM

if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "Cancelled."
    exit 0
fi

# Check dependencies
echo ""
echo "Checking dependencies..."
python3 -c "import sentence_transformers" 2>/dev/null || {
    echo "❌ sentence-transformers not installed"
    echo "Install with: pip install sentence-transformers"
    exit 1
}
echo "✅ Dependencies OK"
echo ""

# 1. Generate alternative embeddings
echo "================================================================================"
echo "PHASE 1: GENERATING ALTERNATIVE EMBEDDINGS"
echo "================================================================================"
echo ""

python3 -m src.embedding.alternative_embeddings \
    --prompts-file data/prompts/prompts.jsonl \
    --output-dir data/processed/alternative_embeddings \
    --models openai-large mpnet

echo ""

# 2. Robustness analysis
echo "================================================================================"
echo "PHASE 2: ROBUSTNESS ANALYSIS"
echo "================================================================================"
echo ""

python3 -m src.evaluation.embedding_robustness \
    --results-file results/v3/multi_model/all_models_results.csv \
    --alt-embeddings-dir data/processed/alternative_embeddings \
    --output-dir results/v3/robustness

echo ""
echo "================================================================================"
echo "ROBUSTNESS ANALYSIS COMPLETE!"
echo "================================================================================"
echo ""
echo "Output:"
echo "  📊 Results: results/v3/robustness/embedding_robustness_results.csv"
echo "  📈 Figure:  results/v3/robustness/embedding_robustness_comparison.png"
echo ""
echo "For your paper (Appendix B):"
echo "  - Report correlations for all 3 embedding models"
echo "  - Include the comparison figure"
echo "  - Cite: 'Results were robust to embedding model choice (see Appendix B)'"
echo ""
echo "================================================================================"
