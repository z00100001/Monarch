import json
from collections import Counter

path = "data/processed/balanced/balanced.json"

label_counter = Counter()

try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for entry in data:
            label = entry.get("label")
            if isinstance(label, str):
                label_counter[label.strip().lower()] += 1
except Exception as e:
    print(f"Error loading {path}: {e}")

print("Label Frequencies (from 'label'):\n")
for label, count in label_counter.most_common():
    print(f"{label}: {count}")
