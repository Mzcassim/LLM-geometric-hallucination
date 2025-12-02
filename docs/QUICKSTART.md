# Quick Reference Guide

**For detailed documentation, see [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)**

---

## Running Experiments

### Master Script
```bash
./run_reproduction.sh
```

**Options:**
1. Single-Model Deep Dive (V2)
2. Multi-Model Consistency (V3) ⭐ Most Important
3. Novel Experiments (Attacks & Steering)

---

## After Experiment Finishes

### 1. Run Analysis
```bash
./run_complete_analysis.sh
```

Generates:
- Statistical tests
- Tables
- Figures

### 2. Verify Judgments
```bash
python3 -m src.evaluation.verify_judgments --n 50
```

Rate 50 random samples manually.

---

## Key Files

### Input
- `data/prompts/prompts.jsonl` - All 538 questions
- `experiments/multi_model_config.yaml` - Model settings

### Output
- `results/v3/multi_model/all_models_results.csv` - Master dataset
- `results/v3/multi_model/figures/*.png` - Visualizations
- `results/v3/multi_model/tables/*.csv` - Tables for paper

---

## Common Commands

### View Results
```bash
# Model performance
cat results/v3/multi_model/tables/table1_model_performance.csv

# Open figures
open results/v3/multi_model/figures/multi_model_risk_manifolds.png
```

### Re-run Just Analysis
```bash
# Statistical tests only
python3 -m src.evaluation.statistical_tests \
    --input-file results/v3/multi_model/all_models_results.csv

# Tables only
python3 -m src.evaluation.generate_tables
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| API key not set | `export OPENAI_API_KEY="..."` |
| Results not found | Wait for experiment to finish |
| Module not found | `pip install -r requirements.txt` |

---

**For Full Details**: See [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)
