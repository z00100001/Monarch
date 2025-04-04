import re

def clean_text(text):
    """
    Cleans and normalizes input text for model training.
    """
    if not isinstance(text, str):
        return ""

    # sets everything to lowercase
    text = text.lower()

    # removes URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # removes special characters and digits
    text = re.sub(r"[^a-zA-Z\s.,!?']", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_dataset(input_path, output_path):
    """
    Loads a JSON dataset and applies cleaning to 'text' field.
    """
    import json
    import os

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        entry["clean_text"] = clean_text(entry.get("text", ""))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Cleaned {len(data)} entries → {output_path}")


if __name__ == "__main__":
    clean_dataset("data/processed/reddit_scored.json", "data/processed/reddit_cleaned.json")
    clean_dataset("data/processed/goemotions_train.json", "data/processed/goemotions_cleaned.json")
