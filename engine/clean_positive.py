import os
import json
import re

# File paths
INPUT_PATH = "data/processed/positive/reddit_positive_20250411_224145.json"
OUTPUT_PATH = "data/processed/positive/reddit_positive_cleaned.json"

def clean_text(text: str) -> str:
    """
    Cleans input text by converting to lowercase, removing URLs, unwanted characters,
    and reducing extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-z\s.,!?']", "", text)            # Remove special characters
    text = re.sub(r"\s+", " ", text)                     # Collapse multiple spaces
    return text.strip()

def load_json(filepath: str):
    """Loads JSON data from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, filepath: str):
    """Saves JSON data to a file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def process_entries(data: list) -> list:
    """Cleans and adds a 'clean_text' field to each entry in the dataset."""
    for entry in data:
        raw_text = f"{entry.get('title', '')} {entry.get('selftext', '')}".strip()
        entry["clean_text"] = clean_text(raw_text)
    return data

def main():
    print(f"📂 Loading data from {INPUT_PATH}")
    data = load_json(INPUT_PATH)

    print("🧼 Cleaning text entries...")
    cleaned_data = process_entries(data)

    print(f"💾 Saving cleaned data to {OUTPUT_PATH}")
    save_json(cleaned_data, OUTPUT_PATH)
    print(f"✅ Done! {len(cleaned_data)} entries processed.")

if __name__ == "__main__":
    main()