"""Early-warning system for hallucination detection."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns


def compute_risk_scores(
    predictor,
    X: np.ndarray,
    scale: bool = True
) -> np.ndarray:
    """
    Compute hallucination risk scores.
    
    Args:
        predictor: Trained predictor model
        X: Feature matrix
        scale: Whether to scale features
        
    Returns:
        Risk scores (probabilities)
    """
    return predictor.predict_proba(X, scale=scale)


def analyze_risk_thresholds(
    risk_scores: np.ndarray,
    y_true: np.ndarray,
    thresholds: List[float] = [0.3, 0.4, 0.5]
) -> pd.DataFrame:
    """
    Analyze performance at different risk thresholds.
    
    Args:
        risk_scores: Risk probabilities
        y_true: True labels
        thresholds: Risk thresholds to analyze
        
    Returns:
        DataFrame with metrics per threshold
    """
    results = []
    
    for threshold in thresholds:
        # Flag top X% as high risk
        n_flag = int(len(risk_scores) * threshold)
        flagged_indices = np.argsort(risk_scores)[-n_flag:]
        
        flagged = np.zeros(len(risk_scores), dtype=bool)
        flagged[flagged_indices] = True
        
        # Compute metrics
        tp = (flagged & (y_true == 1)).sum()
        fp = (flagged & (y_true == 0)).sum()
        tn = (~flagged & (y_true == 0)).sum()
        fn = (~flagged & (y_true == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'pct_flagged': threshold * 100,
            'n_flagged': int(flagged.sum()),
            'n_hallucinations_caught': int(tp),
            'pct_hallucinations_caught': (recall * 100) if (tp + fn) > 0 else 0,
            'precision': precision,
            'recall': recall,
            'fpr': fpr
        })
    
    return pd.DataFrame(results)


def plot_roc_curve(
    risk_scores: np.ndarray,
    y_true: np.ndarray,
    save_path: Optional[Path] = None
):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, risk_scores)
    
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Hallucination Detection')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_precision_recall_curve(
    risk_scores: np.ndarray,
    y_true: np.ndarray,
    save_path: Optional[Path] = None
):
    """Plot precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, risk_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkgreen', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Hallucination Detection')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved PR curve to {save_path}")
    else:
        plt.show()
    
    plt.close()


def simulate_mitigation(
    df: pd.DataFrame,
    risk_scores: np.ndarray,
    threshold: float = 0.3
) -> Dict:
    """
    Simulate effect of mitigation on flagged queries.
    
    In practice, this would re-prompt with conservative instructions.
    Here we just analyze the properties of flagged queries.
    
    Args:
        df: DataFrame with questions and labels
        risk_scores: Risk scores
        threshold: Fraction to flag
        
    Returns:
        Analysis of flagged queries
    """
    n_flag = int(len(risk_scores) * threshold)
    flagged_indices = np.argsort(risk_scores)[-n_flag:]
    
    flagged_df = df.iloc[flagged_indices]
    unflagged_df = df.iloc[~np.isin(np.arange(len(df)), flagged_indices)]
    
    analysis = {
        'n_flagged': len(flagged_df),
        'pct_flagged': threshold * 100,
        'flagged_hallucination_rate': flagged_df['is_hallucinated'].mean() * 100,
        'unflagged_hallucination_rate': unflagged_df['is_hallucinated'].mean() * 100,
        'flagged_category_dist': flagged_df['category'].value_counts().to_dict(),
        'intervention_potential': None
    }
    
    # Estimate potential impact
    baseline_rate = df['is_hallucinated'].mean()
    flagged_rate = flagged_df['is_hallucinated'].mean()
    
    if baseline_rate > 0:
        analysis['intervention_potential'] = {
            'baseline_hallucination_rate': baseline_rate * 100,
            'flagged_hallucination_rate': flagged_rate * 100,
            'enrichment_factor': flagged_rate / baseline_rate if baseline_rate > 0 else 0,
            'message': f"Flagged queries are {flagged_rate/baseline_rate:.1f}x more likely to hallucinate"
        }
    
    return analysis


def generate_early_warning_report(
    predictor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    df_test: pd.DataFrame,
    output_dir: Path,
    thresholds: List[float] = [0.3, 0.4, 0.5]
):
    """Generate complete early-warning analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("EARLY-WARNING SYSTEM ANALYSIS")
    print("="*60)
    
    # Compute risk scores
    risk_scores = compute_risk_scores(predictor, X_test, scale=True)
    
    # Analyze thresholds
    threshold_analysis = analyze_risk_thresholds(risk_scores, y_test, thresholds)
    
    print("\n Threshold Analysis:")
    print(threshold_analysis.to_string(index=False))
    
    # Save threshold analysis
    threshold_analysis.to_csv(output_dir / "threshold_analysis.csv", index=False)
    
    # Plot ROC curve
    plot_roc_curve(risk_scores, y_test, save_path=output_dir / "roc_curve.png")
    
    # Plot PR curve
    plot_precision_recall_curve(risk_scores, y_test, save_path=output_dir / "precision_recall_curve.png")
    
    # Mitigation simulation
    for threshold in thresholds:
        print(f"\nMitigation Simulation (threshold={threshold}):")
        analysis = simulate_mitigation(df_test, risk_scores, threshold=threshold)
        
        print(f"  Flagged: {analysis['n_flagged']} queries ({analysis['pct_flagged']:.1f}%)")
        print(f"  Flagged hallucination rate: {analysis['flagged_hallucination_rate']:.1f}%")
        print(f"  Unflagged hallucination rate: {analysis['unflagged_hallucination_rate']:.1f}%")
        
        if analysis['intervention_potential']:
            print(f"  {analysis['intervention_potential']['message']}")
    
    print(f"\nEarly-warning analysis saved to {output_dir}")
    
    return {
        'risk_scores': risk_scores,
        'threshold_analysis': threshold_analysis,
        'output_dir': output_dir
    }
