#!/bin/bash
# Multi-Model Pipeline (Generation -> Judging -> Analysis -> Visualization)
# Runs the complete "Manifold Bends" experiment across multiple models.

set -e

echo "================================================================================"
echo "MULTI-MODEL PIPELINE: START TO FINISH"
echo "================================================================================"
echo ""

# 1. Configuration
# ----------------
N_PROMPTS=${1:-0}  # Default to 0 (all prompts)
CONFIG_FILE="experiments/multi_model_config.yaml"
OUTPUT_DIR="results/v3/multi_model"
JUDGED_DIR="$OUTPUT_DIR/judged"
AGGREGATED_FILE="$OUTPUT_DIR/all_models_results.csv"

# Check API keys
REQUIRED_KEYS=("OPENAI_API_KEY")
OPTIONAL_KEYS=("ANTHROPIC_API_KEY" "TOGETHER_API_KEY")

for key in "${REQUIRED_KEYS[@]}"; do
    if [ -z "${!key}" ]; then
        echo "Error: $key not set"
        exit 1
    fi
done

# 2. Generation Phase (Parallel)
# ------------------------------
echo "PHASE 1: GENERATION (PARALLEL)"
echo "Running models in parallel (grouped by provider)..."

python3 -m src.pipeline.run_parallel_generation \
    --n-prompts "$N_PROMPTS" \
    --output-dir "$OUTPUT_DIR"

echo "✓ Generation complete"
echo ""

# 3. Judging Phase (Consensus)
# ----------------------------
echo "PHASE 2: JUDGING (CONSENSUS)"
echo "Judging responses with Consensus Panel (GPT-5.1 + Claude Opus-4.5 + Llama 4)..."

python3 -m src.pipeline.run_consensus_judging \
    --input-dir "$OUTPUT_DIR" \
    --output-dir "$JUDGED_DIR"

echo "✓ Judging complete"
echo ""

# 4. Aggregation Phase
# --------------------
echo "PHASE 3: AGGREGATION"
echo "Merging results with geometric features..."
python3 -m src.pipeline.aggregate_multi_model_results \
    --input-dir "$JUDGED_DIR" \
    --output-file "$AGGREGATED_FILE"
echo "✓ Aggregation complete"
echo ""

# 5. Visualization Phase
# ----------------------
echo "PHASE 4: VISUALIZATION"
echo "Generating risk manifolds..."
python3 -m src.visualization.multi_model_manifolds \
    --results-file "$AGGREGATED_FILE" \
    --output-dir "$OUTPUT_DIR/figures"
echo "✓ Visualizations saved to $OUTPUT_DIR/figures"
echo ""

# 6. Analysis Phase
# -----------------
echo "PHASE 5: STATISTICAL ANALYSIS"
python3 -m src.evaluation.multi_model_analysis \
    --input-dir "$JUDGED_DIR"
echo "✓ Analysis complete"
echo ""

echo "================================================================================"
echo "PIPELINE COMPLETE!"
echo "================================================================================"
echo "Outputs:"
echo "  - Master Dataset: $AGGREGATED_FILE"
echo "  - Figures:        $OUTPUT_DIR/figures/"
echo "  - Stats:          $OUTPUT_DIR/hallucination_rates.csv"
echo "================================================================================"
