#!/bin/bash
# Master Reproduction Script
# The single entry point to run the entire project.

set -e

echo "================================================================================"
echo "MANIFOLD BENDS: REPRODUCTION SUITE"
echo "================================================================================"
echo "1. Run Single-Model Deep Dive (V2 Pipeline)"
echo "   - Detailed analysis of GPT-4o-mini"
echo "   - Generates the reference geometry and embeddings"
echo ""
echo "2. Run Multi-Model Consistency (V3 Pipeline)"
echo "   - Tests 10+ frontier models (GPT-5, Claude 4.5, Llama 4)"
echo "   - Requires V2 to be run first (for embeddings)"
echo ""
echo "3. Run Novel Experiments (Attacks & Steering)"
echo "   - Adversarial attacks and geometric steering"
echo "================================================================================"
echo ""

read -p "Select option [1/2/3]: " OPTION

if [ "$OPTION" == "1" ]; then
    echo ""
    echo "Starting Single-Model Deep Dive..."
    ./run_v2_pipeline.sh
    
    echo ""
    echo "Running V3 Analysis on Single-Model Results..."
    ./run_v3_pipeline.sh

elif [ "$OPTION" == "2" ]; then
    echo ""
    echo "Starting Multi-Model Consistency Experiment..."
    echo "(This will test 10 frontier models on all prompts - may take 3-4 hours)"
    echo ""
    read -p "Continue? [y/n]: " CONFIRM
    
    if [ "$CONFIRM" == "y" ] || [ "$CONFIRM" == "Y" ]; then
        ./run_multi_model_pipeline.sh
    else
        echo "Cancelled."
        exit 0
    fi

elif [ "$OPTION" == "3" ]; then
    echo ""
    echo "Starting Novel Experiments..."
    # We can reuse run_v3_pipeline.sh but skip the analysis parts if possible,
    # or just run the specific python modules.
    # For simplicity, let's run the modules directly to be clean.
    
    echo "1. Adversarial Attacks"
    python3 -m src.attacks.manifold_attacks
    
    echo ""
    echo "2. Geometric Steering"
    python3 -m src.mitigation.geometric_steering

else
    echo "Invalid option."
    exit 1
fi
