#!/bin/bash
# V3 Master Execution Script
# Runs all V3 analyses and experiments

set -e

echo "================================================================================"
echo "V3 EXPERIMENT PIPELINE - 'When the Manifold Bends, the Model Lies' (Enhanced)"
echo "================================================================================"
echo ""

# Check for OPENAI_API_KEY
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not set"
    exit 1
fi

echo "✓ OpenAI API key found"
echo ""

# Phase 1: Analysis-Only Improvements (No API Calls)
echo "========================================"
echo "PHASE 1: Analysis-Only Improvements"
echo "========================================"

echo "1.1 Within-category geometry analysis..."
python3 -m src.evaluation.within_category_analysis

echo ""
echo "1.2 Conditional model comparison..."
python3 -m src.evaluation.conditional_models

echo ""
echo "1.3 Factual failures analysis..."
python3 -m src.evaluation.factual_edge_cases

echo ""
echo "✓ Phase 1 complete!"
echo ""

# Phase 2: Visualizations
echo "========================================"
echo "PHASE 2: Risk Manifold Visualizations"
echo "========================================"

echo "2.1 Generating UMAP and t-SNE projections..."
python3 -m src.visualization.risk_manifolds

echo ""
echo "✓ Phase 2 complete!"
echo ""

# Phase 3: Novel Contributions (Requires API Calls)
echo "========================================"
echo "PHASE 3: Novel Contributions"
echo "========================================"

# Check if user wants to run API-heavy experiments
read -p "Run adversarial attacks? (requires ~30 API calls) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "3.1 Running adversarial manifold attacks..."
    mkdir -p src/attacks && touch src/attacks/__init__.py
    python3 -m src.attacks.manifold_attacks
else
    echo "3.1 Skipping adversarial attacks"
fi

echo ""

read -p "Run geometric steering experiment? (requires ~50 API calls) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "3.2 Running geometric steering..."
    mkdir -p src/mitigation && touch src/mitigation/__init__.py
    python3 -m src.mitigation.geometric_steering
else
    echo "3.2 Skipping geometric steering"
fi

echo ""
echo "✓ Phase 3 complete!"
echo ""

# Summary
echo "================================================================================"
echo "V3 PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - results/v3/within_category_analysis.json"
echo "  - results/v3/conditional_model_comparison.json"
echo "  - results/v3/factual_failures_analysis.json"
echo "  - results/v3/figures/*.png"
echo "  - results/v3/adversarial_attacks.csv (if run)"
echo "  - results/v3/geometric_steering.csv (if run)"
echo ""
echo "Next steps:"
echo "  1. Review results/v3/ directory"
echo "  2. Check results/v3/figures/ for publication-quality plots"
echo "  3. Read V3_FINAL_REPORT.md for complete findings"
echo ""
echo "================================================================================"
