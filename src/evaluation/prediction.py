"""Predictive modeling for hallucination detection using geometric features."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns


class HallucinationPredictor:
    """Predict hallucination risk using geometric features."""
    
    def __init__(
        self,
        model_type: str = "logistic",
        random_state: int = 42
    ):
        """
        Initialize predictor.
        
        Args:
            model_type: 'logistic' or 'random_forest'
            random_state: Random seed
        """
        self.model_type = model_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        if model_type == "logistic":
            self.model = LogisticRegression(
                random_state=random_state,
                max_iter=1000,
                class_weight='balanced'
            )
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                class_weight='balanced',
                max_depth=10
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_set: str = "combined",
        category_encode: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare feature matrix from dataframe.
        
        Args:
            df: DataFrame with features
            feature_set: 'category_only', 'geometry_only', or 'combined'
            category_encode: Whether to one-hot encode category
            
        Returns:
            (feature_matrix, feature_names)
        """
        geometry_features = ['local_id', 'curvature_score', 'oppositeness_score']
        
        # Add V2 features if available
        if 'density' in df.columns:
            geometry_features.append('density')
        if 'centrality' in df.columns:
            geometry_features.append('centrality')
        
        feature_names = []
        feature_arrays = []
        
        if feature_set in ["category_only", "combined"]:
            if category_encode:
                # One-hot encode category
                categories = pd.get_dummies(df['category'], prefix='cat')
                feature_arrays.append(categories.values)
                feature_names.extend(categories.columns.tolist())
            else:
                # Use category as-is (for compatibility)
                pass
        
        if feature_set in ["geometry_only", "combined"]:
            # Add geometry features
            geom_data = df[geometry_features].fillna(0).values
            feature_arrays.append(geom_data)
            feature_names.extend(geometry_features)
        
        if len(feature_arrays) == 0:
            raise ValueError(f"No features selected for feature_set: {feature_set}")
        
        X = np.hstack(feature_arrays) if len(feature_arrays) > 1 else feature_arrays[0]
        
        return X, feature_names
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        scale: bool = True
    ):
        """Fit the model."""
        if scale:
            X_train = self.scaler.fit_transform(X_train)
        
        self.model.fit(X_train, y_train)
        
        return self
    
    def predict(self, X: np.ndarray, scale: bool = True) -> np.ndarray:
        """Predict binary labels."""
        if scale:
            X = self.scaler.transform(X)
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray, scale: bool = True) -> np.ndarray:
        """Predict probabilities."""
        if scale:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)[:, 1]  # Probability of hallucination
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        scale: bool = True
    ) -> Dict:
        """
        Evaluate model performance.
        
        Returns:
            Dictionary with metrics
        """
        y_pred = self.predict(X_test, scale=scale)
        y_prob = self.predict_proba(X_test, scale=scale)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'n_samples': len(y_test),
            'n_positive': int(y_test.sum()),
            'n_negative': int(len(y_test) - y_test.sum())
        }
        
        return metrics
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance scores."""
        if self.model_type == "logistic":
            # Coefficients for logistic regression
            importance = np.abs(self.model.coef_[0])
        elif self.model_type == "random_forest":
            # Feature importances for random forest
            importance = self.model.feature_importances_
        else:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df


def train_and_evaluate(
    df: pd.DataFrame,
    feature_sets: List[Dict],
    model_types: List[str] = ["logistic", "random_forest"],
    test_size: float = 0.3,
    val_size: float = 0.15,
    random_state: int = 42,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Train and evaluate models with different feature sets.
    
    Args:
        df: DataFrame with features and labels
        feature_sets: List of dicts with 'name' and 'type' keys
        model_types: Types of models to try
        test_size: Test set fraction
        val_size: Validation set fraction (from remaining after test split)
        random_state: Random seed
        output_dir: Directory to save results
        
    Returns:
        Dictionary with all results
    """
    # Binary hallucination label
    y = df['is_hallucinated'].values
    
    # Split data
    indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y
    )
    
    y_train_val = y[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size/(1-test_size),
        random_state=random_state, stratify=y_train_val
    )
    
    results = {
        'feature_sets': {},
        'best_model': None,
        'best_auc': 0.0
    }
    
    for fs_config in feature_sets:
        fs_name = fs_config.get('name', 'unnamed')
        fs_type = fs_config.get('type', 'combined')
        
        print(f"\n{'='*60}")
        print(f"Feature Set: {fs_name} ({fs_type})")
        print(f"{'='*60}")
        
        results['feature_sets'][fs_name] = {}
        
        for model_type in model_types:
            print(f"\nModel: {model_type}")
            
            # Prepare features
            predictor = HallucinationPredictor(model_type=model_type, random_state=random_state)
            X, feature_names = predictor.prepare_features(df, feature_set=fs_type)
            
            X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
            y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
            
            # Train
            predictor.fit(X_train, y_train, scale=True)
            
            # Evaluate on validation set
            val_metrics = predictor.evaluate(X_val, y_val, scale=True)
            print(f"  Validation AUC: {val_metrics['auc']:.3f}")
            print(f"  Validation Accuracy: {val_metrics['accuracy']:.3f}")
            
            # Evaluate on test set
            test_metrics = predictor.evaluate(X_test, y_test, scale=True)
            print(f"  Test AUC: {test_metrics['auc']:.3f}")
            print(f"  Test Accuracy: {test_metrics['accuracy']:.3f}")
            print(f"  Test F1: {test_metrics['f1']:.3f}")
            
            # Feature importance
            importance_df = predictor.get_feature_importance(feature_names)
            
            # Store results
            results['feature_sets'][fs_name][model_type] = {
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'feature_importance': importance_df.to_dict('records'),
                'model': predictor
            }
            
            # Track best model
            if val_metrics['auc'] > results['best_auc']:
                results['best_auc'] = val_metrics['auc']
                results['best_model'] = {
                    'feature_set': fs_name,
                    'model_type': model_type,
                    'predictor': predictor,
                    'test_auc': test_metrics['auc']
                }
    
    # Save results if output directory provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_results(results, output_dir)
    
    return results


def save_results(results: Dict, output_dir: Path):
    """Save prediction results and plots."""
    # Save metrics table
    metrics_rows = []
    for fs_name, fs_results in results['feature_sets'].items():
        for model_type, model_results in fs_results.items():
            test_m = model_results['test_metrics']
            metrics_rows.append({
                'feature_set': fs_name,
                'model': model_type,
                'test_auc': test_m['auc'],
                'test_accuracy': test_m['accuracy'],
                'test_precision': test_m['precision'],
                'test_recall': test_m['recall'],
                'test_f1': test_m['f1']
            })
    
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "model_comparison.csv", index=False)
    print(f"\nSaved model comparison to {output_dir / 'model_comparison.csv'}")
    
    # Save feature importance for best model
    if results['best_model']:
        best = results['best_model']
        fs_name = best['feature_set']
        model_type = best['model_type']
        importance = results['feature_sets'][fs_name][model_type]['feature_importance']
        
        importance_df = pd.DataFrame(importance)
        importance_df.to_csv(output_dir / "best_model_feature_importance.csv", index=False)
        
        print(f"\nBest model: {fs_name} + {model_type}")
        print(f"Test AUC: {best['test_auc']:.3f}")
