# V2 Pipeline - Quick Start Guide

## Prerequisites

1. **Set API Key**
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

2. **Install Dependencies**
```bash
pip3 install -r requirements.txt
```

---

## Running the Pipeline

### Test Mode (Fast - 40 prompts)
```bash
./run_v2_pipeline.sh --test
```
- Uses 10 prompts per category (40 total)
- Completes in ~10-15 minutes
- Good for testing the system

### Production Mode (Full - 368 prompts)
```bash
./run_v2_pipeline.sh
```
- Uses full benchmark (368 prompts)
- Completes in ~2-4 hours
- Generates publication-ready results

---

## What the Pipeline Does

1. ✅ **Builds Benchmark** - Generates prompts from templates
2. ✅ **Builds Reference Corpus** - Creates normalization framework (one-time)
3. ✅ **Generates Responses** - Gets model answers (3 samples per prompt)
4. ✅ **Judges Responses** - Evaluates hallucinations with LLM-as-a-judge
5. ✅ **Computes Geometry** - Calculates all geometric features
6. ✅ **Aggregates Results** - Merges everything into one dataset
7. ✅ **Runs Predictions** - Trains models and generates early-warning analysis

---

## Output Files

After completion, check:

```
results/
├── all_results.csv              # Complete dataset
├── prediction/
│   ├── model_comparison.csv     # Performance metrics
│   └── best_model_feature_importance.csv
├── early_warning/
│   ├── roc_curve.png           # ROC analysis
│   ├── precision_recall_curve.png
│   └── threshold_analysis.csv
├── figures/                     # Visualizations
└── tables/                      # Summary stats
```

---

## Troubleshooting

**"OPENAI_API_KEY not set"**
```bash
export OPENAI_API_KEY="sk-..."
```

**"Reference corpus not found"**
- Pipeline will build it automatically on first run
- Takes ~2-5 minutes with API calls

**Script permission denied**
```bash
chmod +x run_v2_pipeline.sh
```

---

## Next Steps

1. Run test mode first to verify everything works
2. Run production mode overnight for full results
3. Review `results/prediction/model_comparison.csv`
4. Check early-warning ROC curves
5. Write your paper!

---

## Individual Steps (if needed)

If you want to run steps individually:

```bash
# 1. Build benchmark
python3 -m src.pipeline.build_benchmark_v2 --prompts-per-category 120

# 2. Reference corpus
python3 -m src.geometry.reference_corpus

# 3. Generation
python3 -m src.pipeline.run_generation --config experiments/config_v2.yaml

# 4. Judging
python3 -m src.pipeline.run_judging --config experiments/config_v2.yaml

# 5. Geometry
python3 -m src.pipeline.compute_geometry --config experiments/config_v2.yaml

# 6. Aggregate
python3 -m src.pipeline.aggregate_results --config experiments/config_v2.yaml

# 7. Predictions (see run_v2_pipeline.sh for full command)
```
