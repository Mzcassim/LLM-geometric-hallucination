"""Deduplicate judged files - keep first occurrence of each ID."""

import json
from pathlib import Path
from collections import OrderedDict

judged_dir = Path("results/v3/multi_model/judged")
backup_dir = Path("results/v3/multi_model/judged_backup")

# Create backup
backup_dir.mkdir(exist_ok=True)

print("=" * 70)
print("DEDUPLICATING JUDGED FILES")
print("=" * 70)

for judged_file in sorted(judged_dir.glob("judged_answers_*.jsonl")):
    model_name = judged_file.stem.replace("judged_answers_", "")
    
    # Load all records
    records = []
    with open(judged_file, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    # Deduplicate by ID (keep first)
    seen_ids = set()
    unique_records = []
    for record in records:
        rec_id = record['id']
        if rec_id not in seen_ids:
            seen_ids.add(rec_id)
            unique_records.append(record)
    
    duplicates = len(records) - len(unique_records)
    
    print(f"\n{model_name}:")
    print(f"  Original: {len(records)} records")
    print(f"  Unique:   {len(unique_records)} records")
    print(f"  Removed:  {duplicates} duplicates")
    
    if duplicates > 0:
        # Backup original
        backup_file = backup_dir / judged_file.name
        with open(backup_file, 'w') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"  Backed up to: {backup_file}")
        
        # Write deduplicated
        with open(judged_file, 'w') as f:
            for record in unique_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"  ✅ Deduplicated file saved")
    else:
        print(f"  ✅ No duplicates found")

print("\n" + "=" * 70)
print("DEDUPLICATION COMPLETE")
print("=" * 70)
