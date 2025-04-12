import os
import sys
import json
import re


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# file paths
INPUT_FILE = "data/processed/positive/reddit_positive_cleaned.json"
OUTPUT_FILE = "data/processed/positive/reddit_positive_scored.json"
LEXICON_PATH = "data/raw/external/positive_lexicons.txt"


def load_positive_lexicon():
    lexicon = {}
    with open(LEXICON_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                word, score = parts
                try:
                    score = int(score)
                    if score == 1:  # this keeps all positive words
                        lexicon[word.lower()] = score
                except ValueError:
                    continue
    return lexicon

def score_text_positivity(text, lexicon):
    words = re.findall(r"\b\w+\b", text.lower())
    matches = [w for w in words if w in lexicon]
    score = len(matches) 
    return {
        "score": score,
        "matches": matches
    }


def main():
    print("Loading cleaned dataset...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    lexicon = load_positive_lexicon()
    print("Scoring each post for positivity...")

    for entry in data:
        text = entry.get("clean_text", "")
        result = score_text_positivity(text, lexicon)
        entry["positivity_score"] = result["score"]
        entry["positive_matches"] = result["matches"]

        
        if "labels" not in entry or not entry["labels"]:
            entry["labels"] = ["joy"]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Scored entries saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()