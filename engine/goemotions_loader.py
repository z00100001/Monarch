import os
import json
import csv

GOEMOTIONS_DIR = "data/raw/external/goemotions"
OUTPUT_FILE = "data/processed/goemotions_train.json"

def load_emotion_labels():
    """Loads emotion index → label name from emotions.txt"""
    label_path = os.path.join(GOEMOTIONS_DIR, "emotions.txt")
    with open(label_path, "r") as f:
        return [line.strip() for line in f.readlines()]

def load_ekman_mapping():
    """Loads emotion → ekman category mapping"""
    path = os.path.join(GOEMOTIONS_DIR, "ekman_mapping.json")
    with open(path, "r") as f:
        return json.load(f)

def load_tsv(filepath):
    """Loads a GoEmotions TSV file and returns rows"""
    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        return list(reader)

def parse_goemotions(tsv_rows, emotion_labels, ekman_map=None):
    """Parses TSV rows and maps labels to emotions and Ekman categories"""
    parsed = []

    for row in tsv_rows:
        if len(row) < 3:
            continue 
        text, labels_str, _ = row
        label_ids = list(map(int, labels_str.split(",")))

        emotions = [emotion_labels[i] for i in label_ids]
        ekman = set()
        if ekman_map:
            for e in emotions:
                mapped = ekman_map.get(e, "other")
                if isinstance(mapped, list):
                    ekman.update(mapped)
                else:
                    ekman.add(mapped)

        parsed.append({
            "text": text,
            "labels": emotions,
            "ekman_labels": list(ekman)
        })

    return parsed

def save_to_json(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

def main():
    print("Loading GoEmotions training data...")

    tsv_path = os.path.join(GOEMOTIONS_DIR, "train.tsv")
    emotion_labels = load_emotion_labels()
    ekman_map = load_ekman_mapping()
    rows = load_tsv(tsv_path)
    parsed = parse_goemotions(rows, emotion_labels, ekman_map)

    save_to_json(parsed, OUTPUT_FILE)
    print(f"Saved {len(parsed)} cleaned entries to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
