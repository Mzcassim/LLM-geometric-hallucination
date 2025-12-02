"""Analyze judge confidence and agreement patterns.

This script examines:
1. Confidence distributions per model
2. Judge agreement rates (unanimous vs majority vote)
3. Low-confidence judgments
4. Disagreement patterns (do certain prompts cause disagreement?)
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.io import read_jsonl

from src.utils.io import read_jsonl


def analyze_judge_confidence(judged_dir, output_dir):
    """Analyze judge confidence and agreement."""
    
    judged_path = Path(judged_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load all judged files
    files = list(judged_path.glob("judged_answers_*.jsonl"))
    
    all_records = []
    confidence_stats = []
    agreement_stats = []
    
    for f in files:
        model_name = f.stem.replace('judged_answers_', '')
        records = read_jsonl(f)
        
        model_confidences = []
        unanimous_count = 0
        majority_count = 0
        split_count = 0
        
        for record in records:
            confidence = record.get('judge_confidence', 0)
            model_confidences.append(confidence)
            
            # Check agreement if individual judgments are stored
            if 'individual_judgments' in record:
                judges = record['individual_judgments']
                labels = [j['label'] for j in judges]
                label_counts = Counter(labels)
                
                if len(label_counts) == 1:
                    # Unanimous (3/3)
                    unanimous_count += 1
                    agreement_type = 'unanimous'
                elif max(label_counts.values()) >= 2:
                    # Majority (2/3)
                    majority_count += 1
                    agreement_type = 'majority'
                else:
                    # Split (1/1/1)
                    split_count += 1
                    agreement_type = 'split'
                
                record['agreement_type'] = agreement_type
            
            record['model_name'] = model_name
            all_records.append(record)
        
        # Model-level stats
        confidence_stats.append({
            'model': model_name,
            'mean_confidence': np.mean(model_confidences),
            'median_confidence': np.median(model_confidences),
            'min_confidence': np.min(model_confidences),
            'max_confidence': np.max(model_confidences),
            'std_confidence': np.std(model_confidences)
        })
        
        total = len(records)
        if total > 0:
            agreement_stats.append({
                'model': model_name,
                'unanimous_count': unanimous_count,
                'majority_count': majority_count,
                'split_count': split_count,
                'unanimous_rate': unanimous_count / total,
                'majority_rate': majority_count / total,
                'split_rate': split_count / total
            })
    
    # Save tables
    confidence_df = pd.DataFrame(confidence_stats)
    confidence_df.to_csv(output_path / 'confidence_summary.csv', index=False)
    print(f"Saved confidence summary to {output_path / 'confidence_summary.csv'}")
    
    agreement_df = pd.DataFrame(agreement_stats)
    agreement_df.to_csv(output_path / 'agreement_summary.csv', index=False)
    print(f"Saved agreement summary to {output_path / 'agreement_summary.csv'}")
    
    # Full dataset
    full_df = pd.DataFrame(all_records)
    
    # Identify low-confidence cases
    low_conf_threshold = 0.5
    low_conf = full_df[full_df['judge_confidence'] < low_conf_threshold]
    low_conf[['model_name', 'question', 'model_answer', 'judge_label', 'judge_confidence']].to_csv(
        output_path / 'low_confidence_cases.csv', index=False
    )
    print(f"Found {len(low_conf)} low-confidence cases (<{low_conf_threshold})")
    
    # Visualizations
    create_confidence_visualizations(confidence_df, agreement_df, full_df, output_path)
    
    # Summary report
    print("\n=== CONFIDENCE ANALYSIS SUMMARY ===")
    print(f"\nOverall Confidence:")
    print(f"  Mean: {full_df['judge_confidence'].mean():.3f}")
    print(f"  Median: {full_df['judge_confidence'].median():.3f}")
    print(f"  Std: {full_df['judge_confidence'].std():.3f}")
    
    if len(agreement_df) > 0:
        print(f"\nAverage Agreement Rates:")
        print(f"  Unanimous (3/3): {agreement_df['unanimous_rate'].mean():.1%}")
        print(f"  Majority (2/3): {agreement_df['majority_rate'].mean():.1%}")
        print(f"  Split (1/1/1): {agreement_df['split_rate'].mean():.1%}")
    
    return confidence_df, agreement_df, full_df


def create_confidence_visualizations(confidence_df, agreement_df, full_df, output_path):
    """Create visualizations."""
    
    # Figure 1: Confidence distribution by model
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    ax = axes[0]
    models = full_df['model_name'].unique()
    data = [full_df[full_df['model_name'] == m]['judge_confidence'].values for m in models]
    ax.boxplot(data, labels=models, vert=True)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Judge Confidence', fontsize=12)
    ax.set_title('Judge Confidence Distribution by Model', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    # Agreement rates
    ax = axes[1]
    if len(agreement_df) > 0:
        x = np.arange(len(agreement_df))
        width = 0.25
        
        ax.bar(x - width, agreement_df['unanimous_rate'], width, label='Unanimous (3/3)', color='green', alpha=0.8)
        ax.bar(x, agreement_df['majority_rate'], width, label='Majority (2/3)', color='orange', alpha=0.8)
        ax.bar(x + width, agreement_df['split_rate'], width, label='Split (1/1/1)', color='red', alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Rate', fontsize=12)
        ax.set_title('Judge Agreement Patterns', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(agreement_df['model'], rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'judge_confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {output_path / 'judge_confidence_analysis.png'}")
    
    # Figure 2: Confidence vs Hallucination Rate (only if we have hallucination data)
    if 'is_hallucinated' not in full_df.columns:
        print("Note: Skipping confidence vs hallucination plot (no hallucination labels in data)")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Bin by confidence
    bins = np.linspace(0, 1, 11)
    full_df['conf_bin'] = pd.cut(full_df['judge_confidence'], bins)
    
    # Calculate hallucination rate per bin
    grouped = full_df.groupby('conf_bin', observed=True).agg({
        'is_hallucinated': ['mean', 'count']
    }).reset_index()
    
    bin_centers = [(b.left + b.right) / 2 for b in grouped['conf_bin']]
    hall_rates = grouped['is_hallucinated']['mean']
    counts = grouped['is_hallucinated']['count']
    
    ax.scatter(bin_centers, hall_rates, s=counts*2, alpha=0.6, color='darkblue')
    ax.set_xlabel('Judge Confidence', fontsize=12)
    ax.set_ylabel('Hallucination Rate', fontsize=12)
    ax.set_title('Confidence vs Hallucination Rate', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'confidence_vs_hallucination.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {output_path / 'confidence_vs_hallucination.png'}")


def integrate_human_verification(full_df, human_verification_file, output_path):
    """Integrate human verification data to analyze AI judge calibration."""
    
    print("\n=== HUMAN VERIFICATION INTEGRATION ===")
    
    try:
        with open(human_verification_file, 'r') as f:
            human_data = json.load(f)
        
        human_labels = human_data.get('human_labels', [])
        ai_labels = human_data.get('ai_labels', [])
        
        if not human_labels or not ai_labels:
            print("No human verification data found in file")
            return
        
        # Create dataframe for analysis
        verification_df = pd.DataFrame({
            'human_label': human_labels,
            'ai_label': ai_labels,
            'agreement': [h == a for h, a in zip(human_labels, ai_labels)]
        })
        
        # Overall agreement
        agreement_rate = verification_df['agreement'].mean()
        print(f"\nOverall AI-Human Agreement: {agreement_rate:.1%}")
        
        # Agreement by confidence (if we can match records)
        # Note: This is simplified - in practice, would need to match by ID
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(human_labels, ai_labels)
        
        labels = ['Correct', 'Partial', 'Hallucinated', 'Refused']
        cm_df = pd.DataFrame(cm, index=labels[:cm.shape[0]], columns=labels[:cm.shape[1]])
        cm_df.to_csv(output_path / 'human_ai_confusion_matrix.csv')
        print(f"Saved confusion matrix to {output_path / 'human_ai_confusion_matrix.csv'}")
        
        # Visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('AI Judge vs Human Expert\nConfusion Matrix', fontsize=14, fontweight='bold')
        ax.set_ylabel('Human Label', fontsize=12)
        ax.set_xlabel('AI Judge Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path / 'human_ai_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved confusion matrix visualization")
        
        # Calibration analysis
        print("\n=== CALIBRATION INSIGHTS ===")
        print("(Note: Full calibration by confidence requires matching sampled records)")
        print(f"Agreement Rate: {agreement_rate:.1%}")
        print(f"Total Verified: {len(human_labels)}")
        
    except FileNotFoundError:
        print(f"Human verification file not found: {human_verification_file}")
    except Exception as e:
        print(f"Error integrating human verification: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--judged-dir", default="results/v3/multi_model/judged")
    parser.add_argument("--output-dir", default="results/v3/multi_model/judge_analysis")
    parser.add_argument("--human-verification", default=None, help="Optional: Path to human verification JSON")
    
    args = parser.parse_args()
    
    confidence_df, agreement_df, full_df = analyze_judge_confidence(args.judged_dir, args.output_dir)
    
    # Integrate human verification if provided
    if args.human_verification:
        integrate_human_verification(full_df, args.human_verification, Path(args.output_dir))
    
    print("\n✅ Judge confidence analysis complete!")



if __name__ == "__main__":
    main()
