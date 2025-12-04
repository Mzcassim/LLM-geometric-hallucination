"""Create clean, deduplicated prompts.jsonl from all sources."""

import json
from pathlib import Path
from collections import OrderedDict

prompts_dir = Path("data/prompts")
backup_dir = prompts_dir / "backup_originals"
backup_dir.mkdir(exist_ok=True)

# Load all prompts from all files
all_prompts = []
for jsonl_file in sorted(prompts_dir.glob("*.jsonl")):
    if jsonl_file.name == "prompts.jsonl":
        continue  # Skip the consolidated file
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                all_prompts.append(json.loads(line))

print(f"Loaded {len(all_prompts)} total prompts from source files")

# Deduplicate by ID (keep first)
seen_ids = set()
unique_prompts = []
for prompt in all_prompts:
    prompt_id = prompt['id']
    if prompt_id not in seen_ids:
        seen_ids.add(prompt_id)
        unique_prompts.append(prompt)

print(f"After deduplication: {len(unique_prompts)} unique prompts")
print(f"Removed: {len(all_prompts) - len(unique_prompts)} duplicates")

# Write clean prompts.jsonl
output_file = prompts_dir / "prompts_deduplicated.jsonl"
with open(output_file, 'w') as f:
    for prompt in unique_prompts:
        f.write(json.dumps(prompt, ensure_ascii=False) + '\n')

print(f"\n✅ Wrote clean file: {output_file}")
print(f"   Total unique prompts: {len(unique_prompts)}")

# Move old files to backup
print(f"\nBacking up original files to {backup_dir}/")
for jsonl_file in prompts_dir.glob("*.jsonl"):
    if jsonl_file.name in ["prompts_deduplicated.jsonl"]:
        continue
    backup_path = backup_dir / jsonl_file.name
    jsonl_file.rename(backup_path)
    print(f"  Moved: {jsonl_file.name}")

# Rename clean file to prompts.jsonl
final_path = prompts_dir / "prompts.jsonl"
output_file.rename(final_path)
print(f"\n✅ Final file: {final_path} ({len(unique_prompts)} prompts)")
