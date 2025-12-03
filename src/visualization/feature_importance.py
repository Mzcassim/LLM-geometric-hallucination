import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import os

def generate_feature_importance(data_path, output_path):
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Features to analyze
    features = ['local_id', 'curvature_score', 'density', 'centrality']
    target = 'is_hallucinated'
    
    # Check if columns exist
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns {missing_cols}")
        return

    # Prepare data
    X = df[features]
    y = df[target]
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Train Random Forest
    print("Training Random Forest classifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_imputed, y)
    
    # Get feature importance
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("Feature Importance:")
    print(feature_importance_df)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Geometric Feature Importance for Predicting Hallucinations')
    plt.xlabel('Importance (Gini Impurity)')
    plt.ylabel('Geometric Feature')
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Feature importance plot saved to {output_path}")

if __name__ == "__main__":
    data_path = "results/v3/multi_model/all_models_results.csv"
    output_path = "results/v3/multi_model/figures/feature_importance.png"
    
    # Ensure paths are absolute or relative to project root
    # Assuming script is run from project root
    if not os.path.exists(data_path):
        # Try absolute path based on previous context
        base_path = "/Users/sein/Desktop/homebase/harvard/classes/cs2881/LLM-geometric-hallucination"
        data_path = os.path.join(base_path, data_path)
        output_path = os.path.join(base_path, output_path)
        
    generate_feature_importance(data_path, output_path)
