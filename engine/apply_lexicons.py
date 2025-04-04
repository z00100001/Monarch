import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from engine.lexicons import load_worrywords_lexicon, score_text_anxiety

INPUT_FILE = "data/processed/reddit_clean.json"
OUTPUT_FILE = "data/processed/reddit_scored.json"

def main():
    print("Loading Reddit cleaned data...")
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    lexicon = load_worrywords_lexicon()

    print("Scoring each post for anxiety...")
    for entry in data:
        result = score_text_anxiety(entry["text"], lexicon)
        entry["anxiety_score"] = result["score"]
        entry["anxiety_matches"] = result["matches"]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved scored entries to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
