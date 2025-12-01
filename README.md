# When the Manifold Bends, the Model Lies

**Geometric Predictors of Hallucination in Large Language Models**

This repository implements a research pipeline to investigate the relationship between geometric properties of embedding spaces and hallucination behavior in LLMs. The core hypothesis is that local geometric features (intrinsic dimension, curvature, and "oppositeness") in the embedding manifold may correlate with a model's tendency to hallucinate.

## 📋 Overview

The project:
1. **Builds a Hallucination Benchmark** with four categories of questions designed to elicit varying hallucination behaviors
2. **Generates Model Responses** using a target LLM
3. **Judges Hallucination Severity** using an LLM-as-a-judge approach
4. **Computes Geometric Metrics** around query embeddings (local intrinsic dimension, curvature proxy, geometric oppositeness)
5. **Analyzes Correlations** between geometry and hallucination

## 🚀 Quick Start (V2 Pipeline)

### Installation

```bash
# Clone the repository
cd manifold-bends-model-lies

# Install dependencies
pip install -r requirements.txt

# Set up your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### Running the Full Experiment

We provide a master script to run the entire pipeline end-to-end:

```bash
# Run the full production pipeline (368 prompts)
./run_v2_pipeline.sh

# Run a fast test mode (40 prompts)
./run_v2_pipeline.sh --test
```

The pipeline automatically:
1. Builds the benchmark
2. Generates model responses (multi-sample)
3. Judges hallucinations
4. Computes all geometric features
5. Aggregates results
6. Trains predictive models & generates early-warning analysis

## 📁 Repository Structure

```
manifold-bends-model-lies/
├── README.md                    # This file
├── run_v2_pipeline.sh           # Master executable script
├── ANALYSIS_RESULTS.md          # Summary of latest results
├── requirements.txt             # Python dependencies
├── experiments/
│   ├── config_v2.yaml          # Production configuration
│   └── config_example.yaml     # Test configuration
├── src/
│   ├── config.py               # Configuration management
│   ├── models/                 # Model clients (Embedding, Generation, Judge)
│   ├── geometry/               # Geometric feature computation
│   │   ├── density.py             # Local density estimation
│   │   ├── centrality.py          # Distance to center
│   │   ├── reference_corpus.py    # Normalization corpus builder
│   │   ├── neighbors.py           # k-NN utilities
│   │   ├── intrinsic_dimension.py # TwoNN estimator
│   │   ├── curvature.py           # PCA-based curvature proxy
│   │   └── oppositeness.py        # Geometric oppositeness metric
│   ├── evaluation/             # Evaluation and analysis
│   │   ├── prediction.py          # Predictive modeling (ML)
│   │   ├── early_warning.py       # Early-warning system
│   │   ├── metrics.py             # Correlation and stats
│   │   └── robustness.py          # Robustness checks
│   ├── pipeline/               # End-to-end pipeline scripts
│   └── utils/                  # Utilities
├── data/
│   ├── prompts/                # Benchmark questions (JSONL)
│   ├── processed/              # Intermediate data files
│   └── reference_corpus/       # Normalization data
└── results/
    ├── all_results.csv         # Final merged dataset
    ├── prediction/             # Model performance metrics
    ├── early_warning/          # ROC curves and risk analysis
    ├── figures/                # Generated plots
    └── tables/                 # Analysis tables
```

## 🎯 Benchmark Categories (V2)

The V2 benchmark (368 prompts) includes:

1. **Impossible Questions** (30) - Unsolved problems or logical impossibilities
2. **Nonexistent Entities** (120) - Fabricated people, books, theorems
3. **Ambiguous Questions** (120) - Questions with no single ground truth
4. **Factual Questions** (98) - Clear factual questions (control group)

## 📊 Geometric Features

### 1. Local Density (New in V2)
Inverse average distance to nearest neighbors. Measures how "crowded" or supported a region is.
- **Hypothesis:** Low density (sparse regions) → High hallucination risk.

### 2. Centrality (New in V2)
Distance from the global center of the embedding space.
- **Hypothesis:** High distance (peripheral regions) → High hallucination risk.

### 3. Local Intrinsic Dimension (TwoNN)
Estimates the dimensionality of the manifold near each point.
- **Hypothesis:** High dimension (complex regions) → Confusion.

### 4. Curvature Proxy
PCA residual variance in local neighborhoods.
- **Hypothesis:** High curvature (irregular regions) → Interpolation errors.

### 5. Geometric Oppositeness
Distance from sign-flipped PCA projection to nearest real embedding.
- **Hypothesis:** High oppositeness (extreme/boundary regions) → Hallucination.

## 🔮 Predictive Modeling & Early Warning

The V2 pipeline includes a machine learning module (`src.evaluation.prediction`) that:
1. Trains classifiers (Logistic Regression, Random Forest) to predict hallucinations based on geometry.
2. Generates an **Early Warning Report** identifying risky queries before generation.
3. Simulates mitigation strategies (e.g., "Flagging top 30% of queries catches 85% of lies").

## 🔧 Configuration

Edit `experiments/config_v2.yaml` to customize:

```yaml
# Project settings
project_name: "manifold-bends-v2"

# Benchmark settings
benchmark:
  prompts_per_category: 120

# Generation settings
generation:
  models:
    - name: "gpt-4o-mini"
      samples_per_prompt: 3

# Geometry settings
geometry:
  metrics:
    - "local_id"
    - "curvature"
    - "oppositeness"
    - "density"
    - "centrality"
```

## 📈 Output Files

After running the full pipeline, you'll have:

- `data/prompts/*.jsonl` - Benchmark questions
- `data/processed/model_answers.jsonl` - Generated answers
- `data/processed/judged_answers.jsonl` - Hallucination judgments
- `data/processed/question_embeddings.npy` - Embeddings
- `data/processed/geometry_features.csv` - Geometric metrics
- `results/all_results.csv` - **Complete merged dataset**
- `results/figures/` - Visualizations
- `results/tables/` - Correlation and statistics tables

## 🔬 Analysis

The analysis script generates:

1. **Distribution plots** - Geometry feature distributions by hallucination status
2. **Scatter plots** - Geometry vs. hallucination with correlation coefficients
3. **Box plots** - Geometry features by question category
4. **Bin analysis** - Hallucination rates across geometry feature bins
5. **Correlation tables** - Pearson and Spearman correlations
6. **Descriptive statistics** - Summary stats by category

## 🧪 Example Analysis Workflow

```python
import pandas as pd
from src.evaluation.metrics import compute_correlations

# Load results
df = pd.read_csv("results/all_results.csv")

# Compute correlations
geometry_features = ['local_id', 'curvature_score', 'oppositeness_score']
correlations = compute_correlations(df, geometry_features, target='is_hallucinated')
print(correlations)

# Filter to high-confidence judgments
high_conf = df[df['judge_confidence'] >= 0.7]

# Analyze specific category
impossible = df[df['category'] == 'impossible']
print(impossible[['question', 'judge_label', 'local_id']].head())
```

## 📝 Key Components

### LLM-as-a-Judge Prompt

The judge evaluates answers using a structured prompt that provides:
- The question
- The model's answer
- Ground truth/evidence

It returns:
- **Label**: 0=correct, 1=partial, 2=hallucinated, 3=uncertain/refusal
- **Confidence**: Float from 0 to 1
- **Justification**: Brief explanation

### Geometric Metrics Implementation

- **TwoNN**: Uses distance ratios to 1st and 2nd nearest neighbors
- **Curvature**: PCA residual variance in local neighborhoods
- **Oppositeness**: Distance from sign-flipped PCA projection to nearest real point

## 🤝 Contributing

This is a research project. To extend it:

1. Add more benchmark questions in `src/pipeline/build_benchmark.py`
2. Implement additional geometric metrics in `src/geometry/`
3. Try different models by updating the config
4. Add new analysis methods in `src/evaluation/`

## 📄 License

This project is provided for research purposes.

## 🙏 Acknowledgments

Mohamed Zidan Cassim (for making this experiment)

**For questions or issues, please open an issue on GitHub.**
