import json
import os
import re
import unicodedata
from tqdm import tqdm

def infer_label(text, score, matches):
    text = text.lower().strip()

    def contains_any(phrases):
        return any(p in text for p in phrases)

    depression_critical = [
        "suicidal", "want to die", "kill myself", "just want it to end", "can't do this anymore",
        "nothing matters", "mentally dead", "emotionally dead", "feel like a failure",
        "feel empty", "feel broken", "i'm done", "i'm hopeless", "why keep going",
        "no reason to live", "no purpose", "i don't see a way out", "feel like disappearing"
    ]
    if contains_any(depression_critical) or score >= 8:
        return "depression"

    worry_phrases = [
        "anxious", "anxiety", "panic", "overthinking", "nervous", "uncertain future",
        "future scares me", "job security", "financial stress", "ai taking over",
        "overwhelmed", "fear of failure", "what if", "can't relax"
    ]
    if contains_any(worry_phrases) or 5 <= score < 8:
        return "worry"

    sadness_phrases = [
        "lonely", "cry", "miss the past", "nostalgia", "tired of trying", "not happy",
        "lost hope", "feel disconnected", "worthless", "nobody cares", "feel numb",
        "failed", "numbness", "feel unloved", "life has no color", "feel down",
        "wasting time", "no joy", "no energy", "regret"
    ]
    if contains_any(sadness_phrases):
        return "sadness"

    depression_soft = [
        "don't want to be here", "why do i exist", "no reason to wake up", "sleep all day",
        "stay in bed", "can't feel anything", "dissociating", "can't think straight",
        "can't get out of bed", "don't eat", "don't shower", "life is a blur"
    ]
    if contains_any(depression_soft):
        return "depression"

    anger_phrases = [
        "angry", "hate", "rage", "furious", "fed up", "resent", "pissed off",
        "done putting up with", "so mad", "can't stand this", "yelling", "exploding"
    ]
    if contains_any(anger_phrases):
        return "anger"

    return None


def clean_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8", "ignore")
    text = text.lower()

    typo_corrections = {
        r"socia[i1]": "social",
        r"iife": "life",
        r"coiiege": "college",
        r"faiiure": "failure",
        r"peopie": "people",
        r"disiike": "dislike",
        r"famiiy": "family",
        r"iooking": "looking",
        r"ioneiiness": "loneliness",
        r"loneiy": "lonely",
        r"aione": "alone",
        r"taiiking": "talking",
        r"reiatives": "relatives",
        r"cycie": "cycle",
        r"heip": "help",
        r"iack": "lack",
        r"iiving": "living",
        r"oniine": "online",
        r"girlfriendl": "girlfriend",
        r"mereiy": "merely"
    }

    for pattern, replacement in typo_corrections.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r'\bl\b', 'i', text)

    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def process_reddit_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    cleaned_data = []
    for entry in tqdm(raw_data, desc="Processing posts"):
        text = entry.get("selftext") or entry.get("text", "")
        
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("utf-8", "ignore")

        score = entry.get("anxiety_score", 0)
        matches = entry.get("anxiety_matches", [])

        label = infer_label(text, score, matches)
        if label:
            cleaned_data.append({
                "text": text,
                "clean_text": clean_text(text),
                "labels": [label]
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2)

    print(f"[✓] Saved {len(cleaned_data)} labeled entries to {output_path}")

input_file = "data/processed/reddit_scored.json"
output_file = "data/processed/reddit_cleaned.json"

if __name__ == "__main__":
    process_reddit_file(input_file, output_file)
