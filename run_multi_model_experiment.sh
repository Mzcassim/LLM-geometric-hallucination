#!/bin/bash
# Multi-Model Consistency Experiment
# Tests whether geometric hallucination prediction works across different models

set -e

echo "================================================================================"
echo "MULTI-MODEL CONSISTENCY EXPERIMENT"
echo "================================================================================"
echo ""

# Default: run on subset of prompts (fast mode)
N_PROMPTS=${1:-50}  # Default to 50 prompts, pass argument to override

echo "Running with $N_PROMPTS prompts per category"
echo ""

# Check API keys
REQUIRED_KEYS=("OPENAI_API_KEY")
OPTIONAL_KEYS=("ANTHROPIC_API_KEY" "TOGETHER_API_KEY")

for key in "${REQUIRED_KEYS[@]}"; do
    if [ -z "${!key}" ]; then
        echo "Error: $key not set"
        exit 1
    fi
    echo "✓ $key found"
done

for key in "${OPTIONAL_KEYS[@]}"; do
    if [ -z "${!key}" ]; then
        echo "⚠ $key not set (optional provider will be skipped)"
    else
        echo "✓ $key found"
    fi
done

echo ""

# Model list
MODELS=(
    "gpt-4o-mini:openai"           # Frontier (fast)
    "gpt-3.5-turbo:openai"         # Legacy
)

# Add Anthropic if available
if [ -n "$ANTHROPIC_API_KEY" ]; then
    MODELS+=("claude-3-5-haiku:anthropic")  # Frontier
fi

# Add Together if available
if [ -n "$TOGETHER_API_KEY" ]; then
    MODELS+=("llama-3.1-8b:together")       # Open source
    MODELS+=("mixtral-8x7b:together")       # Lesser-known
fi

echo "Testing ${#MODELS[@]} models:"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo ""

# Create output directory
OUTPUT_DIR="results/v3/multi_model"
mkdir -p "$OUTPUT_DIR"

# For each model, run generation and judging
for model_spec in "${MODELS[@]}"; do
    MODEL_KEY=$(echo "$model_spec" | cut -d':' -f1)
    PROVIDER=$(echo "$model_spec" | cut -d':' -f2)
    
    echo "========================================"
    echo "Model: $MODEL_KEY ($PROVIDER)"
    echo "========================================"
    
    # Run generation with this model
    python3 -m src.pipeline.run_multi_model_generation \
        --model-key "$MODEL_KEY" \
        --n-prompts "$N_PROMPTS" \
        --output-dir "$OUTPUT_DIR"
    
    echo ""
done

echo "================================================================================"
echo "ANALYSIS"
echo "================================================================================"

# Analyze cross-model consistency
python3 -m src.evaluation.multi_model_analysis \
    --input-dir "$OUTPUT_DIR"

echo ""
echo "================================================================================"
echo "MULTI-MODEL EXPERIMENT COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
