import pandas as pd
import json

def aggregate_attacks(filepath):
    try:
        df = pd.read_csv(filepath)
        total = len(df)
        # Assuming 'is_hallucinated' is 1 for hallucination
        # Check if 'is_hallucinated' column exists and its values
        if 'is_hallucinated' not in df.columns:
            return "Column 'is_hallucinated' not found"
        
        # Filter for successful attacks (where original was correct (0) and perturbed is hallucinated (1))
        # But wait, the CSV shows 'is_hallucinated' for the perturbed answer.
        # We should check if the attack *caused* a hallucination.
        # Let's assume we care about the rate of hallucinations in the perturbed set.
        hallucination_rate = df['is_hallucinated'].mean()
        
        # Calculate density drop
        avg_density_change = df['density_change'].mean()
        
        return {
            "total_attacks": total,
            "hallucination_rate": hallucination_rate,
            "avg_density_change": avg_density_change
        }
    except Exception as e:
        return str(e)

def aggregate_steering(filepath):
    try:
        df = pd.read_csv(filepath)
        total = len(df)
        success_rate = df['success'].mean()
        avg_risk_reduction = df['risk_reduction'].mean()
        
        return {
            "total_steering_attempts": total,
            "success_rate": success_rate,
            "avg_risk_reduction": avg_risk_reduction
        }
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    attacks_res = aggregate_attacks("results/v3/adversarial_attacks.csv")
    steering_res = aggregate_steering("results/v3/geometric_steering.csv")
    
    print("Adversarial Attacks Summary:")
    print(json.dumps(attacks_res, indent=2))
    print("\nGeometric Steering Summary:")
    print(json.dumps(steering_res, indent=2))
