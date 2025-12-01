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

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd manifold-bends-model-lies

# Install dependencies
pip install -r requirements.txt

# Set up your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### Running the Full Pipeline

```bash
# 1. Build the benchmark (creates prompt files)
python -m src.pipeline.build_benchmark

# 2. Generate model answers
python -m src.pipeline.run_generation --config experiments/config_example.yaml

# 3. Judge hallucinations
python -m src.pipeline.run_judging --config experiments/config_example.yaml

# 4. Compute geometry features
python -m src.pipeline.compute_geometry --config experiments/config_example.yaml

# 5. Aggregate all results
python -m src.pipeline.aggregate_results --config experiments/config_example.yaml

# 6. Generate visualizations and analysis
python experiments/notebooks/analysis.py --config experiments/config_example.yaml
```

### Quick Demo (with limited samples)

For a quick demo, edit `experiments/config_example.yaml` and set:
```yaml
max_prompts_per_category: 10  # Use only 10 questions per category
```

## 📁 Repository Structure

```
manifold-bends-model-lies/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── experiments/
│   ├── config_example.yaml     # Configuration file
│   └── notebooks/
│       └── analysis.py         # Analysis and visualization script
├── src/
│   ├── config.py               # Configuration management
│   ├── models/                 # Model client abstractions
│   │   ├── embedding_client.py    # Embedding API wrapper
│   │   ├── generation_client.py   # Generation API wrapper
│   │   └── judge_client.py        # LLM-as-a-judge client
│   ├── geometry/               # Geometric feature computation
│   │   ├── neighbors.py           # k-NN utilities
│   │   ├── intrinsic_dimension.py # TwoNN estimator
│   │   ├── curvature.py           # PCA-based curvature proxy
│   │   └── oppositeness.py        # Geometric oppositeness metric
│   ├── evaluation/             # Evaluation and analysis
│   │   ├── metrics.py             # Correlation and stats
│   │   └── robustness.py          # Robustness checks
│   ├── pipeline/               # End-to-end pipeline scripts
│   │   ├── build_benchmark.py     # Create benchmark dataset
│   │   ├── run_generation.py      # Generate model answers
│   │   ├── run_judging.py         # Judge hallucinations
│   │   ├── compute_geometry.py    # Compute geometric features
│   │   └── aggregate_results.py   # Merge all data
│   └── utils/                  # Utilities
│       ├── io.py                  # File I/O helpers
│       ├── seed.py                # Random seed management
│       └── logging_utils.py       # Logging utilities
├── data/
│   ├── prompts/                # Benchmark questions (JSONL)
│   ├── processed/              # Intermediate data files
│   └── logs/                   # Pipeline logs
└── results/
    ├── all_results.csv         # Final merged dataset
    ├── figures/                # Generated plots
    └── tables/                 # Analysis tables
```

## 🎯 Benchmark Categories

The hallucination benchmark includes four categories:

1. **Impossible Questions** - Questions about unsolved problems or logical impossibilities
   - Example: "State and prove the final solution to the Collatz conjecture."

2. **Nonexistent Entities** - Questions about fabricated people, books, theorems, etc.
   - Example: "Who wrote the book 'The Phantom Chronicles of Eldoria'?"

3. **Ambiguous Questions** - Questions with no single ground truth
   - Example: "Who is the greatest philosopher of all time?"

4. **Factual Questions** - Clear factual questions with known answers (control group)
   - Example: "What is the capital of France?"

## 📊 Geometric Features

### 1. Local Intrinsic Dimension (TwoNN)
Estimates the dimensionality of the manifold near each point using the Two Nearest Neighbors method. Higher values suggest more complex local structure.

### 2. Curvature Proxy
Measures local curvature via PCA residual variance in neighborhoods. Higher values indicate more irregular/curved regions.

### 3. Geometric Oppositeness
Flips principal components and measures distance to the nearest real embedding. Captures how "extreme" or boundary-like a region is.

## 🔧 Configuration

Edit `experiments/config_example.yaml` to customize:

```yaml
# Model configuration
embedding_model: "text-embedding-3-large"
generation_model: "gpt-4o-mini"
judge_model: "gpt-4o-mini"

# Data paths
data_dir: "data"
results_dir: "results"

# Experiment parameters
max_prompts_per_category: 300  # Max questions per category
seed: 42                        # Random seed

# Geometry parameters
n_neighbors_id: 20              # Neighbors for intrinsic dimension
n_neighbors_curvature: 30       # Neighbors for curvature
n_pca_components: 10            # PCA components for oppositeness
n_flip_components: 3            # Components to flip for oppositeness

# API parameters
api_timeout: 60
max_retries: 3
batch_size: 100
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

Based on research into the geometric properties of embedding spaces and their relationship to model behavior.

---

**For questions or issues, please open an issue on GitHub.**
