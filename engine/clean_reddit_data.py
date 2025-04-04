import os
import json
import re

RAW_DIR = "data/raw"
OUTPUT_FILE = "data/processed/reddit_clean.json"

def clean_text(text):
    
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s.,!?']", "", text) # filters
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_posts(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    cleaned = []
    for post in data:
        raw_text = post.get("selftext", "") or post.get("title", "")
        if not raw_text or raw_text in ["[removed]", "[deleted]"]:
            continue

        text = clean_text(raw_text)
        if len(text.split()) < 5:
            continue

        cleaned.append({
            "text": text,
            "subreddit": post.get("subreddit", "unknown")
        })

    return cleaned

def process_all_scrapes():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    all_clean = []

    for file in os.listdir(RAW_DIR):
        if file.startswith("reddit_") and file.endswith(".json"):
            print(f"Processing: {file}")
            path = os.path.join(RAW_DIR, file)
            posts = extract_posts(path)
            all_clean.extend(posts)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_clean, f, indent=2)

    print(f"\nSaved {len(all_clean)} cleaned Reddit entries to: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_all_scrapes()
