"""Remove ground truth errors from all dataset files.

Ground truth errors (actually real places, not nonexistent):
1. The Sapphire Coast - Real tourism region in NSW, Australia
2. Lake Meridian - Real lake in Washington State, USA

These were mislabeled as nonexistent entities but are actually real.
All models "hallucinated" because the ground truth was wrong.
"""

import json
from pathlib import Path

# IDs to remove (need to find exact IDs)
ERROR_KEYWORDS = [
    "Sapphire Coast",
    "Lake Meridian"
]

def should_remove(prompt_data):
    """Check if prompt contains error keywords."""
    question = prompt_data.get('question', '')
    entity = prompt_data.get('entity', '')
    
    for keyword in ERROR_KEYWORDS:
        if keyword.lower() in question.lower() or keyword.lower() in entity.lower():
            return True
    return False

# Process prompts.jsonl
prompts_file = Path("data/prompts/prompts.jsonl")
prompts = []
removed_ids = []

print("=" * 70)
print("REMOVING GROUND TRUTH ERRORS")
print("=" * 70)

with open(prompts_file, 'r') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            if should_remove(data):
                removed_ids.append(data['id'])
                print(f"Removing: {data['id']} - {data['question']}")
            else:
                prompts.append(data)

print(f"\nRemoved {len(removed_ids)} prompts with ground truth errors")
print(f"Remaining: {len(prompts)} prompts")

# Save cleaned prompts
with open(prompts_file, 'w') as f:
    for prompt in prompts:
        f.write(json.dumps(prompt, ensure_ascii=False) + '\n')

print(f"✅ Saved cleaned prompts.jsonl")

# Remove from all judged files
judged_dir = Path("results/v3/multi_model/judged")
for judged_file in judged_dir.glob("judged_answers_*.jsonl"):
    records = []
    with open(judged_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if data['id'] not in removed_ids:
                    records.append(data)
    
    with open(judged_file, 'w') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    model_name = judged_file.stem.replace("judged_answers_", "")
    print(f"✅ Cleaned {model_name}: {len(records)} remaining")

# Remove from answer files
answers_dir = Path("results/v3/multi_model")
for answer_file in answers_dir.glob("answers_*.jsonl"):
    records = []
    with open(answer_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if data['id'] not in removed_ids:
                    records.append(data)
    
    with open(answer_file, 'w') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    model_name = answer_file.stem.replace("answers_", "")
    print(f"✅ Cleaned {model_name} answers: {len(records)} remaining")

print("\n" + "=" * 70)
print(f"GROUND TRUTH ERRORS REMOVED: {len(removed_ids)} prompts")
print(f"IDs removed: {', '.join(removed_ids)}")
print("=" * 70)
