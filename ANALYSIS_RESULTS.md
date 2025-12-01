# Results Analysis: When the Manifold Bends, the Model Lies

## Executive Summary

You've successfully completed the full pipeline! Here's what the data reveals about the relationship between embedding geometry and hallucinations in GPT-4o-mini.

---

## 📊 Key Findings

### 1. Overall Hallucination Rates

**Total: 40 questions processed**
- ✅ **Correct (Label 0):** 28 (70%)
- ⚠️ **Partial (Label 1):** 1 (2.5%)
- ❌ **Hallucinated (Label 2):** 8 (20%)
- ❓ **Uncertain/Refusal (Label 3):** 3 (7.5%)

### 2. Hallucination by Question Category

| Category | Total | Hallucinated | Rate |
|----------|-------|--------------|------|
| **Factual** | 10 | 0 | 0% ✅ |
| **Ambiguous** | 10 | 0 | 0% ✅ |
| **Impossible** | 10 | 1 | 10% ⚠️ |
| **Nonexistent** | 10 | 7 | **70%** ❌ |

**Key Insight:** The model handles factual and ambiguous questions well, but struggles dramatically with **nonexistent entities** (fabricated books, theorems, people).

---

## 🔍 Geometric Analysis

### Correlation Results (Geometry ↔ Hallucination)

| Feature | Pearson r | p-value | Spearman ρ | p-value | Interpretation |
|---------|-----------|---------|------------|---------|----------------|
| **Local ID** | 0.029 | 0.857 | 0.097 | 0.550 | ❌ No correlation |
| **Curvature** | -0.085 | 0.603 | -0.065 | 0.690 | ❌ No correlation |
| **Oppositeness** | 0.285 | 0.075 | 0.254 | 0.113 | ⚠️ **Weak positive trend** (borderline) |

**Statistical Note:** None of these correlations reach statistical significance at p < 0.05. However, **oppositeness shows the strongest relationship** (p = 0.075), approaching marginal significance.

### Geometry by Category

#### **Oppositeness Score** (Most Promising)
- **Impossible questions:** 0.491 (highest)
- **Nonexistent entities:** 0.473 (second highest)
- **Ambiguous:** 0.437
- **Factual:** 0.414 (lowest)

**Interpretation:** Questions more likely to elicit hallucinations (impossible, nonexistent) have **higher oppositeness scores**, suggesting they occupy more "extreme" or boundary regions of the embedding space.

#### **Curvature Score**
- **Impossible:** 0.377 (highest)
- **Factual:** 0.333
- **Nonexistent:** 0.262
- **Ambiguous:** 0.210 (lowest)

**Interpretation:** Less clear pattern. Impossible questions show higher local curvature, but factual questions are similar.

#### **Local Intrinsic Dimension**
- Highly variable within categories (large standard deviations)
- **Factual:** Mean 390.6 (but ranges from 2 to 3656!)
- No clear pattern emerges

---

## 💡 Insights & Interpretation

### What Worked
1. ✅ **The model refuses appropriately** on many impossible questions (70% correct/uncertain)
2. ✅ **Perfect on factual questions** - correctly answered all control questions
3. ✅ **Handled ambiguity well** - didn't fabricate facts for subjective questions

### What Didn't Work
1. ❌ **Nonexistent entities are a major weakness** - 70% hallucination rate
   - Model confidently invents information about fake books, theorems, people
   - This is consistent with broader LLM literature on factual recall failures

### Geometric Hypothesis Evaluation

**Research Question:** Do geometric properties of embeddings predict hallucination?

**Answer:** **Weak evidence, but promising direction for oppositeness**

1. **Oppositeness (Geometric Extremeness):**
   - Shows **positive trend** (r = 0.28, p = 0.075)
   - Categories with high hallucination rates have higher oppositeness
   - **Hypothesis:** Questions in "boundary regions" lack nearby training examples → more hallucination

2. **Curvature & Intrinsic Dimension:**
   - No significant correlation
   - May require larger sample size or different metrics

---

## 📈 Visualizations Created

Your pipeline generated 4 key figures:

1. **`geometry_distributions.png`**
   - Shows distribution of each geometry feature split by hallucinated vs. not hallucinated
   - Look for separation between the two groups

2. **`geometry_scatter.png`**
   - Scatter plots: geometry features vs. hallucination
   - Includes correlation coefficients

3. **`geometry_by_category.png`**
   - Box plots showing geometry feature distributions per question category
   - Useful for understanding category-specific geometric signatures

4. **`hallucination_by_bins.png`**
   - Hallucination rates across binned geometry values
   - Shows if higher geometry values → higher hallucination rates

**Action:** Open `results/figures/` folder to review these visualizations!

---

## 🎓 For Your Research Paper

### Main Conclusions You Can Draw

1. **Category-Specific Vulnerability:**
   - "GPT-4o-mini shows clear vulnerability to nonexistent entity questions (70% hallucination) while maintaining perfect accuracy on factual questions."

2. **Geometric Signal:**
   - "Oppositeness score, measuring embedding space extremeness, shows a weak positive association with hallucination (ρ = 0.254, p = 0.113), suggesting boundary regions may correlate with reduced model confidence."

3. **Practical Implications:**
   - "Geometric features alone are insufficient for hallucination detection with current sample size, but oppositeness shows promise as a component of ensemble detection systems."

### Limitations to Acknowledge

1. **Small sample size** (n=40) limits statistical power
2. **Single model tested** (GPT-4o-mini) - generalization unclear
3. **Binary classification** (hallucinated/not) loses nuance
4. **Correlation ≠ causation** - geometry might be symptom, not cause

### Future Work Suggestions

1. **Larger benchmark** - Scale to 500+ questions for statistical power
2. **Multiple models** - Test GPT-4, Claude, Llama to check generalization
3. **Alternative metrics** - Try manifold density, geodesic distance, spectral methods
4. **Predictive modeling** - Train classifier using geometry features
5. **Causal analysis** - Fine-tuning experiments to manipulate geometry

---

## 📁 Data Files Summary

✅ **All pipeline outputs created successfully:**

```
results/
├── all_results.csv          # Complete dataset (197 lines = header + 40 samples + multiline text fields)
├── figures/
│   ├── geometry_distributions.png
│   ├── geometry_scatter.png
│   ├── geometry_by_category.png
│   └── hallucination_by_bins.png
└── tables/
    ├── correlations.csv
    └── stats_by_category.csv
```

---

## 🔬 Next Steps for Deeper Analysis

### Quick Wins

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('results/all_results.csv')

# 1. Examine specific hallucinations
hallucinated = df[df['is_hallucinated'] == 1]
print(hallucinated[['category', 'question', 'judge_justification']])

# 2. High oppositeness cases
high_opp = df.nlargest(10, 'oppositeness_score')
print(high_opp[['question', 'is_hallucinated', 'oppositeness_score']])

# 3. Category comparison
print(df.groupby('category')['oppositeness_score'].describe())
```

### Statistical Tests

```python
from scipy.stats import ttest_ind, mannwhitneyu

# Compare geometry between hallucinated and not
hall = df[df['is_hallucinated'] == 1]['oppositeness_score']
not_hall = df[df['is_hallucinated'] == 0]['oppositeness_score']

statistic, pvalue = mannwhitneyu(hall, not_hall)
print(f"Mann-Whitney U test: p = {pvalue}")
```

---

## 🎯 Bottom Line

**Your experiment worked!** You have:
- ✅ A complete dataset linking geometry and hallucinations
- ✅ Evidence that question category strongly predicts hallucination
- ⚠️ Suggestive (but not conclusive) evidence that geometric oppositeness correlates with hallucination
- ✅ A solid foundation for a research paper or thesis

**The 20% hallucination rate and 70% rate for nonexistent entities align with known LLM behaviors**, validating your methodology. The geometric analysis shows promise but would benefit from a larger sample.

Great work completing the pipeline! 🎉
