import os
import json

# Paths to your datasets
POSITIVE_PATH = "data/processed/positive/reddit_positive_scored.json"
NEGATIVE_PATH = "data/processed/reddit_scored.json"
OUTPUT_PATH = "data/processed/goemotions_cleaned.json"

def load_data(path, label):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = []
    for entry in data:
        clean_text = entry.get("clean_text")
        if not clean_text:
            continue
        result.append({
            "text": clean_text,
            "labels": [label],
            "ekman_labels": ["other"]
        })
    return result

def main():
    print("📥 Loading positive and negative datasets...")
    pos_data = load_data(POSITIVE_PATH, "joy")
    neg_data = load_data(NEGATIVE_PATH, "distress")

    all_data = pos_data + neg_data
    print(f"✅ Combined entries: {len(all_data)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)

    print(f"💾 Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()