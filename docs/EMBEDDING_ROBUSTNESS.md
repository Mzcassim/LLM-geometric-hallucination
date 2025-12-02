# Embedding Robustness Analysis (Optional)

## Overview

This optional analysis tests whether the geometry-hallucination correlation is **robust to embedding model choice**.

It compares results across:
1. **text-embedding-3-small** (Original, 1536-dim, OpenAI)
2. **text-embedding-3-large** (3072-dim, OpenAI)
3. **all-mpnet-base-v2** (768-dim, Open Source)

---

## When to Run This

- ✅ **After** completing the main multi-model experiment
- ✅ When preparing a journal submission (for Appendix B)
- ✅ To strengthen reproducibility claims

---

## Prerequisites

### Install Sentence Transformers
```bash
pip install sentence-transformers
```

### Ensure Main Results Exist
```bash
# Check that this file exists:
ls results/v3/multi_model/all_models_results.csv
```

---

## Running the Analysis

### One Command
```bash
./run_embedding_robustness.sh
```

This will:
1. Generate embeddings with OpenAI large + Sentence Transformers
2. Compute geometry for each embedding
3. Compare correlations across all 3 models

**Time**: ~10-15 minutes
**Cost**: ~$0.10 (for OpenAI large embeddings)

---

## Outputs

### Table
`results/v3/robustness/embedding_robustness_results.csv`

| embedding_model | geometry_feature | correlation | p_value |
|----------------|------------------|-------------|---------|
| text-embedding-3-small | curvature_score | 0.XX | <0.001 |
| text-embedding-3-large | curvature_score | 0.XX | <0.001 |
| all-mpnet-base-v2 | curvature_score | 0.XX | <0.001 |

### Figure
`results/v3/robustness/embedding_robustness_comparison.png`

Bar chart showing correlations for each geometry feature across the 3 embedding models.

---

## In Your Paper (Appendix B)

### Text Template

> **Appendix B: Robustness to Embedding Model**
>
> To ensure our findings were not dependent on a specific embedding model, we replicated the analysis using two alternative embeddings: OpenAI's text-embedding-3-large (3072 dimensions) and the open-source all-mpnet-base-v2 model (768 dimensions). 
>
> Results were consistent across all three models (see Table B1). The correlation between curvature and hallucination rate remained significant (r = 0.XX, p < 0.001 for text-embedding-3-small; r = 0.XX, p < 0.001 for text-embedding-3-large; r = 0.XX, p < 0.001 for all-mpnet-base-v2). 
>
> This demonstrates that our geometric signatures are robust to embedding model choice, including open-source alternatives that enhance reproducibility.

### Include Figure
- **Figure B1**: `embedding_robustness_comparison.png`
- **Caption**: "Geometry-hallucination correlations across three embedding models. Error bars represent 95% confidence intervals. Results were robust to embedding choice."

---

## Advanced Usage

### Generate Only Specific Embeddings
```bash
# OpenAI large only
python3 -m src.embedding.alternative_embeddings --models openai-large

# Open source only
python3 -m src.embedding.alternative_embeddings --models mpnet
```

### Run Analysis Manually
```bash
# After generating embeddings
python3 -m src.evaluation.embedding_robustness \
    --results-file results/v3/multi_model/all_models_results.csv \
    --alt-embeddings-dir data/processed/alternative_embeddings \
    --output-dir results/v3/robustness
```

---

## Notes

- **Not Required**: This is optional. Your main results with text-embedding-3-small are valid.
- **Strengthens Paper**: Reviewers appreciate robustness checks.
- **Open Source Option**: Including all-mpnet-base-v2 makes your work more reproducible.

---

**Back to Main Documentation**: [README.md](../README.md)
