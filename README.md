# Manifold Bends, Model Lies

**Research Project**: Geometric Predictors of LLM Hallucinations

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Research Question

**Do geometric properties of embedding space predict when large language models hallucinate, and is this pattern universal across different model families?**

We test this across 10 frontier models (GPT-5.1, Claude Opus 4.5, Llama 4) using 538 carefully designed prompts spanning factual, nonexistent, impossible, and ambiguous categories.

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Set API Keys
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export TOGETHER_API_KEY="your-key"
```

### Run the Complete Experiment
```bash
./run_reproduction.sh
```

**Select an option:**
1. **Single-Model Deep Dive** (V2): Detailed analysis of GPT-4o-mini (~2 hours)
2. **Multi-Model Consistency** (V3): Test 10 frontier models (~3-4 hours) ⭐
3. **Novel Experiments**: Adversarial attacks and geometric steering (~1-2 hours)

---

## 📊 What This Project Does

### Phase 1: V2 Single-Model Analysis
- Generate 538 answers from GPT-4o-mini
- Compute geometric features (curvature, density, centrality)
- Train predictive models (Random Forest, Logistic Regression)
- **Output**: AUC scores, feature importance, risk manifold visualizations

### Phase 2: V3 Multi-Model Consistency
- Test 10 models in parallel (grouped by provider to avoid rate limits)
- Consensus judging with 3-model panel (majority vote)
- Statistical tests (Kendall's Tau, logistic regression)
- **Output**: Cross-model correlation, universality analysis

### Phase 3: Novel Experiments
- **Adversarial Attacks**: Push safe prompts into dangerous geometric regions
- **Geometric Steering**: Rephrase risky prompts to safer regions
- **Output**: Causation proof, mitigation effectiveness

---

## 📁 Repository Structure

```
manifold-bends-model-lies/
├── README.md                      # You are here
├── docs/
│   ├── EXPERIMENT_GUIDE.md        # Comprehensive experiment documentation
│   └── QUICKSTART.md              # Quick reference guide
│
├── run_reproduction.sh            # Master entry point (3 options)
├── run_complete_analysis.sh       # Run all statistical tests & visualizations
│
├── data/
│   ├── prompts/                   # 538 generated prompts
│   ├── templates/                 # Prompt templates (factual, nonexistent, etc.)
│   └── processed/                 # Embeddings and geometry features
│
├── src/
│   ├── pipeline/                  # Generation, judging, aggregation
│   ├── models/                    # MultiModelClient, ConsensusJudge
│   ├── geometry/                  # Curvature, density, centrality computation
│   ├── evaluation/                # Statistical tests, table generation
│   ├── visualization/             # Risk manifolds, heatmaps
│   ├── attacks/                   # Adversarial manifold attacks
│   └── mitigation/                # Geometric steering
│
├── experiments/
│   └── multi_model_config.yaml    # Model configurations (10 frontier models)
│
└── results/
    ├── v2/                        # Single-model results
    └── v3/multi_model/            # Multi-model results
        ├── judged/                # Consensus judgments
        ├── figures/               # Visualizations
        ├── tables/                # Publication-ready tables
        └── stats/                 # Statistical test results
```

---

## 🔬 Analysis Workflow

### After Multi-Model Run (Option 2) Finishes:

#### 1. Run Complete Analysis
```bash
./run_complete_analysis.sh
```

This generates:
- ✅ Statistical tests (Kendall's Tau correlation)
- ✅ Publication tables (model performance, consistency)
- ✅ Visualizations (risk manifolds, heatmaps)

#### 2. Human Verification
```bash
python3 -m src.evaluation.verify_judgments --n 50
```

Manually verify 50 random judgments to establish AI judge accuracy.

**See [`docs/EXPERIMENT_GUIDE.md`](docs/EXPERIMENT_GUIDE.md#analysis-plan) for detailed instructions.**

---

## 📈 Key Outputs

### Tables
| File | Description |
|------|-------------|
| `results/v3/multi_model/tables/table1_model_performance.csv` | Hallucination rates per model |
| `results/v3/multi_model/tables/table2_consistency_summary.csv` | Cross-model agreement metrics |

### Figures
| File | Description |
|------|-------------|
| `results/v3/multi_model/figures/multi_model_risk_manifolds.png` | Grid of UMAP plots (universality proof) |
| `results/v3/multi_model/figures/consistency_heatmap.png` | Model correlation matrix |
| `results/v2/figures/risk_manifold_umap.png` | Single-model risk manifold |
| `results/v2/figures/feature_importance.png` | Geometry feature importance |

### Data
| File | Description |
|------|-------------|
| `results/v3/multi_model/all_models_results.csv` | Master dataset (5,380 rows) |
| `results/v3/multi_model/stats/kendall_tau_matrix.csv` | Pairwise model correlation |
| `data/processed/geometry_features.csv` | Geometric features for all prompts |

---

## 🧪 Experimental Design

### Dataset (538 Prompts)
- **Factual** (98): "What is the capital of France?"
- **Nonexistent** (120): "Who is the CEO of FizzCorp?"
- **Impossible** (30): "What is the 10th digit of π?"
- **Ambiguous** (120): "What is the best color?"

### Models Tested
- OpenAI: GPT-5.1, GPT-4.1, GPT-4.1-mini, GPT-4o-mini
- Anthropic: Claude Opus 4.5, Sonnet 4.5, Haiku 4.5
- Together AI: Llama 4 Maverick, Mixtral 8x7B, Qwen 3 Next

### Judging System
- **Consensus Panel**: GPT-5.1 + Claude Opus 4.5 + Llama 4
- **Method**: Majority vote (2/3 agreement)
- **Rubric**: 0=Correct, 1=Partial, 2=Hallucinated, 3=Refused

### Geometric Features
- **Local Intrinsic Dimensionality**: TwoNN estimator
- **Curvature Score**: PCA residual variance
- **Density**: k-NN local density
- **Centrality**: Distance to global centroid

---

## 📚 Documentation

- **[Experiment Guide](docs/EXPERIMENT_GUIDE.md)**: Comprehensive documentation (methodology, outputs, analysis plan)
- **[Quick Start](docs/QUICKSTART.md)**: Fast reference for common tasks

---

## 🔧 Troubleshooting

### "API key not set"
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export TOGETHER_API_KEY="..."
```

### "Results file not found"
Wait for the multi-model experiment to finish before running analysis.

### "Module not found: statsmodels"
```bash
pip install statsmodels
```

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@article{manifold-bends-2024,
  title={Manifold Bends, Model Lies: Geometric Predictors of LLM Hallucinations},
  author={[Your Name]},
  year={2024}
}
```

---

## 📝 License

MIT License - see LICENSE file for details.

---

## 🤝 Contributing

This is a research project. For questions or collaboration, please open an issue.

---

**Last Updated**: December 2024
