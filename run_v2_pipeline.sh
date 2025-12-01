#!/bin/bash

# Master script to run the complete V2 pipeline
# Usage: ./run_v2_pipeline.sh [--test]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TEST_MODE=false
if [ "$1" == "--test" ]; then
    TEST_MODE=true
    echo -e "${YELLOW}Running in TEST MODE (small dataset)${NC}"
    CONFIG_FILE="experiments/config_example.yaml"
    PROMPTS_PER_CATEGORY=10
else
    echo -e "${GREEN}Running in PRODUCTION MODE (full dataset)${NC}"
    CONFIG_FILE="experiments/config_v2.yaml"
    PROMPTS_PER_CATEGORY=120
fi

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY environment variable not set${NC}"
    echo "Please run: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

echo -e "${GREEN}✓ OpenAI API key found${NC}"
echo ""

# Function to print step header
print_step() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Function to check if step succeeded
check_success() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1 completed successfully${NC}"
    else
        echo -e "${RED}✗ $1 failed${NC}"
        exit 1
    fi
}

# Start pipeline
echo -e "${GREEN}Starting V2 Pipeline...${NC}"
echo "Timestamp: $(date)"
echo ""

# Step 1: Build Benchmark
print_step "Step 1/7: Building Benchmark ($PROMPTS_PER_CATEGORY prompts/category)"
python3 -m src.pipeline.build_benchmark_v2 --prompts-per-category $PROMPTS_PER_CATEGORY --seed 42
check_success "Benchmark generation"

# Step 2: Build Reference Corpus (if needed)
print_step "Step 2/7: Building Reference Corpus"
if [ ! -d "data/reference_corpus" ]; then
    echo "Reference corpus not found, building..."
    python3 -m src.geometry.reference_corpus
    check_success "Reference corpus building"
else
    echo -e "${YELLOW}Reference corpus already exists, skipping...${NC}"
fi

# Step 3: Generate Model Responses
print_step "Step 3/7: Generating Model Responses"
python3 -m src.pipeline.run_generation --config $CONFIG_FILE
check_success "Model response generation"

# Step 4: Judge Responses
print_step "Step 4/7: Judging Responses with LLM-as-a-Judge"
python3 -m src.pipeline.run_judging --config $CONFIG_FILE
check_success "Response judging"

# Step 5: Compute Geometry Features
print_step "Step 5/7: Computing Geometric Features"
python3 -m src.pipeline.compute_geometry --config $CONFIG_FILE
check_success "Geometry computation"

# Step 6: Aggregate Results
print_step "Step 6/7: Aggregating Results"
python3 -m src.pipeline.aggregate_results --config $CONFIG_FILE
check_success "Results aggregation"

# Step 7: Run Predictive Modeling
print_step "Step 7/7: Running Predictive Modeling & Early Warning Analysis"
python3 << 'EOF'
from src.evaluation.prediction import train_and_evaluate
from src.evaluation.early_warning import generate_early_warning_report
import pandas as pd
from pathlib import Path

print("Loading aggregated results...")
df = pd.read_csv('results/all_results.csv')

print(f"Dataset: {len(df)} samples")
print(f"Hallucination rate: {df['is_hallucinated'].mean()*100:.1f}%")

# Define feature sets
feature_sets = [
    {'name': 'category_only', 'type': 'category_only'},
    {'name': 'geometry_only', 'type': 'geometry_only'},
    {'name': 'combined', 'type': 'combined'}
]

# Train and evaluate models
print("\nTraining predictive models...")
results = train_and_evaluate(
    df, 
    feature_sets, 
    model_types=['logistic', 'random_forest'],
    output_dir=Path('results/prediction')
)

# Get best model
best = results['best_model']
print(f"\nBest Model: {best['feature_set']} + {best['model_type']}")
print(f"Validation AUC: {results['best_auc']:.3f}")
print(f"Test AUC: {best['test_auc']:.3f}")

# Run early warning analysis
print("\nGenerating early-warning analysis...")
predictor = best['predictor']

# Split for early warning (reuse test set from training)
from sklearn.model_selection import train_test_split
y = df['is_hallucinated'].values
_, test_idx = train_test_split(
    range(len(df)), test_size=0.3, random_state=42, stratify=y
)

X, feature_names = predictor.prepare_features(df, feature_set=best['feature_set'])
X_test = X[test_idx]
y_test = y[test_idx]
df_test = df.iloc[test_idx]

generate_early_warning_report(
    predictor, X_test, y_test, df_test,
    output_dir=Path('results/early_warning'),
    thresholds=[0.3, 0.4, 0.5]
)

print("\n✓ All analyses complete!")
EOF

check_success "Predictive modeling and early warning"

# Summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Pipeline Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results saved to:"
echo "  - results/all_results.csv (complete dataset)"
echo "  - results/prediction/ (model performance)"
echo "  - results/early_warning/ (risk analysis)"
echo "  - results/figures/ (visualizations)"
echo "  - results/tables/ (summary statistics)"
echo ""
echo "Next steps:"
echo "  1. Review results in results/"
echo "  2. Check model_comparison.csv for performance"
echo "  3. View ROC curves in results/early_warning/"
echo "  4. Write your paper! 📝"
echo ""
echo "Pipeline completed at: $(date)"
