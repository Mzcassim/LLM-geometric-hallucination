"""Test script to demonstrate new consensus confidence calculation."""

from collections import Counter

def calculate_old_confidence(individual_results):
    """Old method: Average of individual confidences."""
    return sum(r['confidence'] for r in individual_results) / len(individual_results)

def calculate_new_confidence(individual_results):
    """New method: Agreement rate × average individual confidence."""
    labels = [r['label'] for r in individual_results]
    counts = Counter(labels)
    majority_count = counts.most_common(1)[0][1]
    
    agreement_rate = majority_count / len(labels)
    avg_individual_confidence = sum(r['confidence'] for r in individual_results) / len(individual_results)
    
    return agreement_rate * avg_individual_confidence, agreement_rate

# Example case from user: Sapphire Coast
print("=" * 70)
print("EXAMPLE: Sapphire Coast Prompt (2-1 split)")
print("=" * 70)

individual_judgments = [
    {'label': 2, 'confidence': 0.95, 'judge': 'gpt-5.1'},       # Hallucination
    {'label': 0, 'confidence': 0.95, 'judge': 'claude-opus'},  # Correct (disagrees!)
    {'label': 2, 'confidence': 0.95, 'judge': 'llama-4'}       # Hallucination
]

print("\nIndividual Judgments:")
for j in individual_judgments:
    print(f"  {j['judge']}: Label {j['label']}, Confidence {j['confidence']}")

old_conf = calculate_old_confidence(individual_judgments)
new_conf, agreement = calculate_new_confidence(individual_judgments)

print(f"\nOLD Method (average):  {old_conf:.3f}")
print(f"NEW Method (agreement × avg):  {new_conf:.3f}")
print(f"  → Agreement rate: {agreement:.2f} (2/3 judges)")
print(f"  → Avg individual confidence: 0.95")
print(f"  → Final: {agreement:.2f} × 0.95 = {new_conf:.3f}")

print("\n" + "=" * 70)
print("EXAMPLE: Unanimous Agreement (3-0)")
print("=" * 70)

unanimous = [
    {'label': 2, 'confidence': 0.95, 'judge': 'gpt-5.1'},
    {'label': 2, 'confidence': 0.95, 'judge': 'claude-opus'},
    {'label': 2, 'confidence': 0.95, 'judge': 'llama-4'}
]

print("\nIndividual Judgments:")
for j in unanimous:
    print(f"  {j['judge']}: Label {j['label']}, Confidence {j['confidence']}")

old_conf = calculate_old_confidence(unanimous)
new_conf, agreement = calculate_new_confidence(unanimous)

print(f"\nOLD Method (average):  {old_conf:.3f}")
print(f"NEW Method (agreement × avg):  {new_conf:.3f}")
print(f"  → Agreement rate: {agreement:.2f} (3/3 judges)")
print(f"  → Avg individual confidence: 0.95")
print(f"  → Final: {agreement:.2f} × 0.95 = {new_conf:.3f}")

print("\n" + "=" * 70)
print("KEY IMPROVEMENT:")
print("=" * 70)
print("Split decisions now have LOWER confidence (0.633 vs 0.950)")
print("Unanimous decisions maintain HIGH confidence (0.950)")
print("This better reflects actual certainty in the judgment.")
