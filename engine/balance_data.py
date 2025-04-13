import json
import os
import random
import unicodedata
import re
from collections import defaultdict

goemotions_path = "data/processed/goemotions_cleaned.json"
positive_path = "data/processed/positive/reddit_positive_scored.json"
negative_path = "data/processed/reddit_scored.json"
output_path = "data/processed/balanced/balanced.json"

target_labels = {"joy", "sadness", "anger", "worry", "depression"}

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = text.replace("\\n", " ").replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_and_filter(path):
    worry_keywords = ["anxiety", "anxious", "panic", "worried", "fear", "nervous", "stressed"]

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = []
    for entry in data:
        raw_labels = entry.get("labels", [])
        label = entry.get("label")

        if not label and isinstance(raw_labels, list):
            filtered_labels = [l for l in raw_labels if l in target_labels]
            if len(filtered_labels) == 1:
                label = filtered_labels[0]

        if not label:
            subreddit = entry.get("subreddit", "").lower()
            text_blob = f"{entry.get('text', '')} {entry.get('clean_text', '')}".lower()

            if subreddit == "depression":
                label = "depression"
            elif any(word in text_blob for word in worry_keywords):
                label = "worry"

        if label not in target_labels:
            continue

        text = clean_text(entry.get("text", ""))
        clean = clean_text(entry.get("clean_text", text))

        if not clean:
            continue

        filtered.append({
            "text": text,
            "clean_text": clean,
            "label": label
        })

    return filtered


all_entries = []
for path in [goemotions_path, positive_path, negative_path]:
    if os.path.exists(path):
        print(f"Loaded: {path}")
        all_entries.extend(load_and_filter(path))
    else:
        print(f"Missing file: {path}")

groups = defaultdict(list)
for entry in all_entries:
    groups[entry["label"]].append(entry)

present_labels = set(groups.keys())
missing = target_labels - present_labels
if missing:
    print(f"Missing labels in data: {missing}")
    exit(1)

min_count = min(len(groups[label]) for label in target_labels)
print(f"Balancing all labels to {min_count} entries each")

balanced = []
for label in target_labels:
    balanced.extend(random.sample(groups[label], min_count))

random.shuffle(balanced)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(balanced, f, indent=2)

print(f"Final balanced dataset saved to: {output_path}")
