#!/bin/bash
# Complete Analysis Pipeline
# Runs all statistical tests, generates tables, and creates figures

set -e

echo "================================================================================"
echo "COMPLETE ANALYSIS PIPELINE"
echo "================================================================================"
echo ""

RESULTS_FILE="results/v3/multi_model/all_models_results.csv"
STATS_DIR="results/v3/multi_model/stats"
TABLES_DIR="results/v3/multi_model/tables"
FIGURES_DIR="results/v3/multi_model/figures"

# Check if results exist
if [ ! -f "$RESULTS_FILE" ]; then
    echo "❌ Results file not found: $RESULTS_FILE"
    echo "Run the multi-model pipeline first (./run_reproduction.sh -> Option 2)"
    exit 1
fi

echo "✅ Found results file: $RESULTS_FILE"
echo ""

# 1. Statistical Tests
echo "================================================================================"
echo "PHASE 1: STATISTICAL TESTS"
echo "================================================================================"
echo ""
echo "Computing Kendall's Tau correlation, logistic regression p-values..."
python3 -m src.evaluation.statistical_tests \
    --input-file "$RESULTS_FILE" \
    --output-dir "$STATS_DIR"

echo ""

# 2. Generate Tables
echo "================================================================================"
echo "PHASE 2: GENERATE TABLES"
echo "================================================================================"
echo ""
echo "Creating publication-ready tables..."
python3 -m src.evaluation.generate_tables \
    --results-file "$RESULTS_FILE" \
    --stats-dir "$STATS_DIR" \
    --output-dir "$TABLES_DIR"

echo ""

# 3. Multi-Model Analysis (hallucination rates, hard prompts)
echo "================================================================================"
echo "PHASE 3: MULTI-MODEL ANALYSIS"
echo "================================================================================"
echo ""
echo "Analyzing hallucination patterns and hard prompts..."
python3 -m src.evaluation.multi_model_analysis \
    --input-dir "results/v3/multi_model/judged"

echo ""

# 4. Visualizations
echo "================================================================================"
echo "PHASE 4: VISUALIZATIONS"
echo "================================================================================"
echo ""
echo "Generating risk manifolds and consistency heatmaps..."
python3 -m src.visualization.multi_model_manifolds \
    --results-file "$RESULTS_FILE" \
    --output-dir "$FIGURES_DIR"

echo ""

# 5. Judge Confidence Analysis
echo "================================================================================"
echo "PHASE 5: JUDGE CONFIDENCE & AGREEMENT"
echo "================================================================================"
echo ""
echo "Analyzing judge confidence and agreement patterns..."

# Check if human verification exists
HUMAN_VERIFICATION="results/v3/multi_model/human_verification_report.json"
if [ -f "$HUMAN_VERIFICATION" ]; then
    echo "  Found human verification data, integrating..."
    python3 -m src.evaluation.judge_confidence_analysis \
        --judged-dir "results/v3/multi_model/judged" \
        --output-dir "results/v3/multi_model/judge_analysis" \
        --human-verification "$HUMAN_VERIFICATION"
else
    python3 -m src.evaluation.judge_confidence_analysis \
        --judged-dir "results/v3/multi_model/judged" \
        --output-dir "results/v3/multi_model/judge_analysis"
fi

echo ""

echo "================================================================================"
echo "ANALYSIS COMPLETE!"
echo "================================================================================"
echo ""
echo "Output Locations:"
echo "  📊 Tables:  $TABLES_DIR/"
echo "  📈 Figures: $FIGURES_DIR/"
echo "  📉 Stats:   $STATS_DIR/"
echo "  ⚖️  Judges:  results/v3/multi_model/judge_analysis/"
echo ""
echo "Next Steps:"
echo "  1. Review tables and figures"
echo "  2. Run human verification: python3 -m src.evaluation.verify_judgments"
echo "  3. Populate paper with results"
echo ""
echo "================================================================================"
