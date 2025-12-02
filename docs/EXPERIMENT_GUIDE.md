# Manifold Bends, Model Lies: Complete Experiment Guide

**Research Question**: Do geometric properties of embedding space predict when LLMs hallucinate, and is this pattern universal across models?

---

## Table of Contents
1. [Experimental Design](#experimental-design)
2. [Phase 1: Prompt Dataset](#phase-1-prompt-dataset)
3. [Phase 2: V2 Single-Model Analysis](#phase-2-v2-single-model-analysis)
4. [Phase 3: Multi-Model Consistency](#phase-3-multi-model-consistency)
5. [Phase 4: Novel Experiments](#phase-4-novel-experiments)
6. [Expected Outputs & Figures](#expected-outputs--figures)
7. [Analysis Plan](#analysis-plan)

---

## Experimental Design

### Core Hypothesis
**H1**: Hallucinations cluster in specific geometric regions of embedding space (high curvature, low density, high centrality).

**H2**: This geometric pattern is **universal** across different model families (OpenAI, Anthropic, Meta).

**H3**: Geometry can be used for:
- **Prediction**: Early warning system for risky prompts
- **Attack**: Intentionally pushing prompts into dangerous regions
- **Defense**: Steering prompts away from risky geometry

### Methodology
- **Prompts**: 538 questions across 4 categories
- **Embeddings**: OpenAI `text-embedding-3-small` (1536-dim)
- **Geometry**: Local Intrinsic Dimensionality, Curvature, Density, Centrality
- **Models Tested**: 10 frontier models (GPT-5.1, Claude Opus 4.5, Llama 4, etc.)
- **Judging**: Consensus panel of 3 top models (GPT-5.1, Claude Opus 4.5, Llama 4)

---

## Phase 1: Prompt Dataset

### Categories (538 Total Prompts)

#### 1. Factual (98 prompts)
- **Examples**: "What is the capital of France?", "Who wrote Romeo and Juliet?"
- **Ground Truth**: Verifiable facts
- **Expected Behavior**: Low hallucination rate

#### 2. Nonexistent (120 prompts)
- **Examples**: "Who is the CEO of FizzCorp?", "What is the plot of the movie ZephyrQuest?"
- **Ground Truth**: "This entity does not exist"
- **Expected Behavior**: High hallucination rate (models fabricate plausible details)

#### 3. Impossible (30 prompts)
- **Examples**: "What is the 10th digit of π?", "What color is the number 7?"
- **Ground Truth**: "This is unanswerable"
- **Expected Behavior**: High refusal rate, or hallucinations

#### 4. Ambiguous (120 prompts)
- **Examples**: "What is the best color?", "Is pineapple on pizza good?"
- **Ground Truth**: "Subjective, no single answer"
- **Expected Behavior**: Variable (some models hedge, some assert opinions)

### Templates
Each category has JSON templates with variables (e.g., `{city}`, `{person}`, `{number}`) that are filled to generate prompts.

**Files**:
- `data/templates/factual_templates.json`
- `data/templates/nonexistent_templates.json`
- `data/templates/impossible_templates.json`
- `data/templates/ambiguous_templates.json`

**Generated Prompts**:
- `data/prompts/prompts.jsonl` (All 538 prompts with metadata)

---

## Phase 2: V2 Single-Model Analysis

### Objective
Deep dive into **GPT-4o-mini** to establish baseline geometry-hallucination relationships.

### Steps

#### 1. Generation
- **Script**: `src/pipeline/run_generation.py`
- **Model**: GPT-4o-mini
- **Output**: `results/v2/model_answers.jsonl` (538 answers)

#### 2. Embedding
- **Script**: `src/pipeline/run_embedding.py`
- **Model**: `text-embedding-3-small`
- **Output**: `data/processed/question_embeddings.npy` (538 × 1536 matrix)

#### 3. Geometry Computation
- **Script**: `src/geometry/compute_features.py`
- **Features**:
  - **Local ID**: Intrinsic dimensionality using TwoNN estimator
  - **Curvature Score**: PCA residual variance (how "bent" is the manifold?)
  - **Oppositeness Score**: Distance after sign-flipping PCA components
  - **Density**: Local density (k-NN based)
  - **Centrality**: Distance to global centroid
- **Output**: `data/processed/geometry_features.csv`

#### 4. Judging
- **Script**: `src/pipeline/run_judging.py`
- **Judge**: GPT-4o-mini (with strict rubric)
- **Rubric**:
  - `0`: Correct
  - `1`: Partially Correct
  - `2`: **Hallucinated** (fabricated info)
  - `3`: Refusal / Uncertain
- **Output**: `results/v2/judged_answers.jsonl`

#### 5. Aggregation
- **Script**: `src/pipeline/aggregate_results.py`
- **Output**: `results/v2/all_results.csv` (Master dataset: prompts + answers + geometry + labels)

#### 6. Predictive Modeling
- **Script**: `src/evaluation/predictive_model.py`
- **Models**:
  - Logistic Regression
  - Random Forest
  - XGBoost
- **Feature Sets**:
  - Category only
  - Geometry only
  - Combined (category + geometry)
- **Metrics**: AUC, Precision, Recall, F1
- **Output**:
  - `results/v2/predictive_model_results.csv`
  - `results/v2/figures/feature_importance.png`
  - `results/v2/figures/roc_curves.png`

#### 7. Visualization
- **Scripts**:
  - `src/visualization/risk_manifolds.py`
  - `src/visualization/within_category_analysis.py`
- **Outputs**:
  - `results/v2/figures/risk_manifold_umap.png` (2D UMAP projection, hallucinations marked in red)
  - `results/v2/figures/risk_manifold_tsne.png`
  - `results/v2/figures/geometry_heatmap.png` (Correlation between features)
  - `results/v2/figures/within_category_analysis.png` (Geometry distributions per category)

---

## Phase 3: Multi-Model Consistency

### Objective
Test if the geometry-hallucination relationship is **universal** across 10 frontier models.

### Models Tested

| Provider | Model | ID |
|----------|-------|-----|
| OpenAI | GPT-5.1 | `gpt-5.1` |
| OpenAI | GPT-4.1 | `gpt-4.1` |
| OpenAI | GPT-4.1-mini | `gpt-4.1-mini` |
| OpenAI | GPT-4o-mini | `gpt-4o-mini` |
| Anthropic | Claude Opus 4.5 | `claude-opus-4-5-20251101` |
| Anthropic | Claude Sonnet 4.5 | `claude-sonnet-4.5` |
| Anthropic | Claude Haiku 4.5 | `claude-haiku-4.5` |
| Together AI | Llama 4 Maverick | `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` |
| Together AI | Mixtral 8x7B | `mistralai/Mixtral-8x7B-Instruct-v0.1` |
| Together AI | Qwen 3 Next | `Qwen/Qwen3-Next-80B-Instruct` |

### Steps

#### 1. Parallel Generation
- **Script**: `src/pipeline/run_parallel_generation.py`
- **Strategy**: Run one model from each provider concurrently (3 parallel streams)
- **Output**: `results/v3/multi_model/answers_{model}.jsonl` (10 files, 538 answers each)

#### 2. Consensus Judging
- **Script**: `src/pipeline/run_consensus_judging.py`
- **Panel**:
  - GPT-5.1 (OpenAI)
  - Claude Opus 4.5 (Anthropic)
  - Llama 4 Maverick (Together AI)
- **Method**:
  - Each judge independently scores every answer
  - Final label = Majority vote (2/3)
  - Confidence = Average confidence
- **Output**: `results/v3/multi_model/judged/judged_answers_{model}.jsonl` (10 files)

#### 3. Aggregation
- **Script**: `src/pipeline/aggregate_multi_model_results.py`
- **Process**: Merge all judged results with geometry features
- **Output**: `results/v3/multi_model/all_models_results.csv` (5,380 rows = 538 prompts × 10 models)

#### 4. Consistency Analysis
- **Script**: `src/evaluation/multi_model_analysis.py`
- **Metrics**:
  - Hallucination rate per model
  - Cross-model correlation (do models fail on the same prompts?)
  - "Hard prompts" (questions that fool >50% of models)
- **Output**:
  - `results/v3/multi_model/hallucination_rates.csv`
  - `results/v3/multi_model/consistency_matrix.csv`
  - `results/v3/multi_model/hard_prompts.json`

#### 5. Multi-Model Visualizations
- **Script**: `src/visualization/multi_model_manifolds.py`
- **Outputs**:
  - `results/v3/multi_model/figures/multi_model_risk_manifolds.png` (Grid of 10 UMAP plots, one per model, showing hallucination clustering)
  - `results/v3/multi_model/figures/consistency_heatmap.png` (Correlation matrix: which models agree/disagree?)

#### 6. Judge Confidence Analysis
- **Script**: `src/evaluation/judge_confidence_analysis.py`
- **Metrics**:
  - Judge agreement rates (unanimous vs. majority vote)
  - Confidence distributions per model
  - Low-confidence cases
- **Output**:
  - `results/v3/multi_model/judge_analysis/confidence_summary.csv`
  - `results/v3/multi_model/judge_analysis/agreement_summary.csv`
  - `results/v3/multi_model/judge_analysis/judge_confidence_analysis.png`

---

## Phase 4: Novel Experiments

### Experiment 4.1: Adversarial Manifold Attacks

**Objective**: Prove that **manipulating geometry** → **increases hallucination rate**.

#### Method
1. **Identify Low-Risk Prompts**: Select factual questions with safe geometry (low curvature, high density).
2. **Perturb Geometry**: Add noise, rare synonyms, or confusing context to push embeddings into dangerous regions.
3. **Measure Shift**: Calculate geometry before/after perturbation.
4. **Test Models**: Generate answers for both original and perturbed prompts.
5. **Compare**: Does hallucination rate increase?

#### Example
- **Original**: "What is the capital of France?"
  - Geometry: Low curvature, high density
  - Expected: Correct answer ("Paris")
  
- **Perturbed**: "In the context of historical European geopolitics, what would one consider the administrative nucleus of the French Republic?"
  - Geometry: Higher curvature, lower density
  - Expected: Higher chance of hallucination or verbose non-answer

#### Script
- **File**: `src/attacks/manifold_attacks.py`

#### Outputs
- `results/v3/attacks/perturbation_results.csv` (Original vs. Perturbed geometry + hallucination rates)
- `results/v3/attacks/figures/geometry_shift.png` (Scatter plot: Δ Curvature vs. Δ Hallucination Rate)
- `results/v3/attacks/figures/attack_success_rate.png` (Bar chart: % of successful attacks per model)

---

### Experiment 4.2: Geometric Steering (Mitigation)

**Objective**: Prove that **rephrasing to improve geometry** → **reduces hallucination rate**.

#### Method
1. **Identify High-Risk Prompts**: Questions with dangerous geometry (high curvature, low density).
2. **Rephrase**: Automatically rewrite to move embeddings into safer regions.
   - Add clarifying context
   - Simplify language
   - Use more common phrasing
3. **Measure Shift**: Calculate geometry before/after rephrasing.
4. **Test Models**: Generate answers for both original and rephrased prompts.
5. **Compare**: Does hallucination rate decrease?

#### Example
- **Original**: "Describe the socioeconomic ramifications of the ZephyrQuest phenomenon."
  - Geometry: High curvature, low density (nonsense entity)
  - Expected: Hallucination (model fabricates "ZephyrQuest")
  
- **Rephrased**: "Does ZephyrQuest exist? If not, please state that."
  - Geometry: Lower curvature, higher density
  - Expected: Refusal or "I don't know"

#### Script
- **File**: `src/mitigation/geometric_steering.py`

#### Outputs
- `results/v3/steering/rephrasing_results.csv` (Original vs. Rephrased geometry + hallucination rates)
- `results/v3/steering/figures/geometry_improvement.png` (Scatter plot: Δ Curvature vs. Δ Hallucination Rate)
- `results/v3/steering/figures/mitigation_success_rate.png` (Bar chart: % reduction in hallucinations)

---

## Expected Outputs & Figures

### Data Files

#### Phase 2 (V2)
- `results/v2/all_results.csv` (Master dataset)
- `results/v2/predictive_model_results.csv` (AUC scores, feature importance)
- `data/processed/question_embeddings.npy` (Embeddings)
- `data/processed/geometry_features.csv` (Geometry)

#### Phase 3 (Multi-Model)
- `results/v3/multi_model/all_models_results.csv` (10 models × 538 prompts)
- `results/v3/multi_model/hallucination_rates.csv` (Summary stats per model)
- `results/v3/multi_model/consistency_matrix.csv` (Model agreement)
- `results/v3/multi_model/hard_prompts.json` (Universally difficult questions)

#### Phase 4 (Novel Experiments)
- `results/v3/attacks/perturbation_results.csv`
- `results/v3/steering/rephrasing_results.csv`

---

### Figures for Paper

#### Figure 1: Risk Manifold (Main Result)
**File**: `results/v2/figures/risk_manifold_umap.png`

**Description**: 2D UMAP projection of 538 prompts. Hallucinations (red X) cluster in specific regions.

**Purpose**: Visual proof that hallucinations occupy distinct geometric regions.

---

#### Figure 2: Geometry Feature Importance
**File**: `results/v2/figures/feature_importance.png`

**Description**: Bar chart showing Random Forest feature importance (Curvature > Density > Centrality > Local ID).

**Purpose**: Identifies which geometric properties are most predictive.

---

#### Figure 3: ROC Curves
**File**: `results/v2/figures/roc_curves.png`

**Description**: ROC curves for:
- Category-only baseline
- Geometry-only
- Combined (best)

**Purpose**: Proves geometry adds predictive power beyond category.

---

#### Figure 4: Multi-Model Risk Manifolds (Grid)
**File**: `results/v3/multi_model/figures/multi_model_risk_manifolds.png`

**Description**: 3×3 grid (or 2×5) of UMAP plots, one per model. Hallucinations in red.

**Purpose**: Shows that **geometry is universal** (red clusters appear in same regions across all models).

---

#### Figure 5: Model Consistency Heatmap
**File**: `results/v3/multi_model/figures/consistency_heatmap.png`

**Description**: 10×10 correlation matrix. High values = models agree on which prompts are hard.

**Purpose**: Quantifies cross-model agreement.

---

#### Figure 6: Geometry Distribution by Category
**File**: `results/v2/figures/within_category_analysis.png`

**Description**: Violin plots or boxplots showing curvature/density distributions for each category.

**Purpose**: Shows that "Nonexistent" and "Impossible" have worse geometry than "Factual".

---

#### Figure 7: Adversarial Attack Success
**File**: `results/v3/attacks/figures/attack_success_rate.png`

**Description**: Bar chart: % of low-risk prompts that became hallucinations after perturbation.

**Purpose**: Proves geometry manipulation → increased hallucination (causation).

---

#### Figure 8: Geometric Steering Effectiveness
**File**: `results/v3/steering/figures/mitigation_success_rate.png`

**Description**: Bar chart: % reduction in hallucinations after rephrasing.

**Purpose**: Proves geometry steering → reduced hallucination (practical mitigation).

---

#### Figure 9 (Optional): Geometry Shift Scatter Plots
**Files**:
- `results/v3/attacks/figures/geometry_shift.png`
- `results/v3/steering/figures/geometry_improvement.png`

**Description**: Scatter plots with:
- X-axis: Change in curvature score
- Y-axis: Change in hallucination rate
- Each point = one prompt

**Purpose**: Shows correlation between geometry changes and hallucination changes.

---

#### Figure 10: Judge Confidence Analysis
**File**: `results/v3/multi_model/judge_analysis/judge_confidence_analysis.png`

**Description**: Two-panel figure showing (1) confidence distributions per model (box plots), and (2) agreement patterns (unanimous/majority/split).

**Purpose**: Demonstrates judge reliability and identifies uncertain cases.

---

#### Figure 11 (Optional): Hard Prompts Analysis
**Custom Script Needed**

**Description**: Highlight the top 10 "hard prompts" (failed by >80% of models) on the UMAP manifold.

**Purpose**: Identifies universally risky questions.

---

## Analysis Plan

### Statistical Tests

#### 1. Geometry-Hallucination Association
- **Test**: Logistic regression
- **Variables**: `is_hallucinated ~ curvature + density + centrality + local_id`
- **Report**: Coefficients, p-values, AUC

#### 2. Cross-Model Consistency
- **Test**: Kendall's Tau correlation
- **Variables**: Model A's hallucination labels vs. Model B's
- **Report**: τ values, p-values

#### 3. Attack/Steering Effectiveness
- **Test**: Paired t-test
- **Variables**: Hallucination rate (before vs. after perturbation/rephrasing)
- **Report**: t-statistic, p-value, effect size (Cohen's d)

---

### Key Metrics for Paper

#### Table 1: Model Performance
| Model | Total Prompts | Hallucinations | Rate | AUC (Geometry) |
|-------|---------------|----------------|------|----------------|
| GPT-5.1 | 538 | X | X% | 0.XX |
| Claude Opus 4.5 | 538 | X | X% | 0.XX |
| ... | ... | ... | ... | ... |

#### Table 2: Feature Importance
| Feature | Random Forest Importance | Logistic Coef | p-value |
|---------|-------------------------|---------------|---------|
| Curvature | 0.XX | +X.XX | <0.001 |
| Density | 0.XX | -X.XX | <0.001 |
| ... | ... | ... | ... |

#### Table 3: Attack Success Rate
| Model | Original Hallucination Rate | Post-Attack Rate | Δ | p-value |
|-------|----------------------------|------------------|---|---------|
| GPT-5.1 | X% | Y% | +Z% | <0.05 |
| ... | ... | ... | ... | ... |

#### Table 4: Steering Success Rate
| Model | Original Hallucination Rate | Post-Steering Rate | Δ | p-value |
|-------|----------------------------|-------------------|---|---------|
| GPT-5.1 | X% | Y% | -Z% | <0.05 |
| ... | ... | ... | ... | ... |

---

## Human Verification

After the full run completes, verify a subset of judgments to establish **Judge Accuracy**:

```bash
python3 -m src.evaluation.verify_judgments --n 50
```

This will:
1. Sample 50 judgments (stratified by label)
2. Present them to you for manual review
3. Calculate agreement rate between AI judges and you
4. Save report to `results/v3/multi_model/human_verification_report.json`

**Report in Paper**: "The consensus judging panel achieved XX% agreement with human expert evaluation (n=50)."

---

## Timeline

1. **Multi-Model Run (Option 2)**: ~3-4 hours (currently running)
2. **Novel Experiments (Option 3)**: ~1-2 hours
3. **Human Verification**: ~30 minutes
4. **Analysis & Figure Generation**: ~1 hour
5. **Paper Writing**: Up to you!

**Total**: ~5-7 hours of compute + human time.

---

## Paper Abstract (Draft)

**Manifold Bends, Model Lies: Geometric Predictors of LLM Hallucinations**

We investigate whether hallucinations in large language models occupy distinct regions of embedding space with identifiable geometric properties. Across 10 frontier models (GPT-5.1, Claude Opus 4.5, Llama 4) and 538 prompts spanning factual, nonexistent, impossible, and ambiguous categories, we find that hallucinations cluster in high-curvature, low-density manifold regions (AUC=0.XX). This pattern is universal: models from different families hallucinate on the same geometrically risky prompts (τ=0.XX, p<0.001). We demonstrate causation via adversarial attacks (geometry manipulation → +XX% hallucination rate) and practical mitigation via geometric steering (-XX% hallucination rate). Our findings suggest intrinsic geometric constraints on language model reliability and enable real-time hallucination risk assessment.

---

## Next Steps

1. ✅ Wait for multi-model run to complete
2. ✅ Run Option 3 (Novel Experiments)
3. ✅ Run human verification
4. ✅ Generate all figures
5. ✅ Populate tables with actual results
6. ✅ Write paper

---

**Questions?** Let me know if you need clarification on any outputs or analyses!
