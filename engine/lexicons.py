import os
import csv
import re

WORRYWORDS_PATH = "data/raw/external/worrywords.txt"

STOPWORDS = set([
    "a", "an", "the", "and", "to", "of", "in", "it", "is", "i", "you", "he", "she",
    "we", "they", "me", "my", "your", "our", "on", "at", "with", "this", "that",
    "for", "as", "was", "are", "be", "but", "so", "do", "just", "have", "has",
    "or", "not", "if", "then", "will", "from", "up", "down", "by"
])

def load_worrywords_lexicon():
    """Load strong anxiety-indicative words from worrywords.txt"""
    lexicon = {}
    with open(WORRYWORDS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            word = row["Term"].strip().lower()
            try:
                score = int(row["OrdinalClass"])
                if score >= 3:  # Only high-anxiety words
                    lexicon[word] = score
            except ValueError:
                continue
    return lexicon

def score_text_anxiety(text, lexicon):
    """Score the input text based on matching worry words"""
    text = text.lower()
    words = re.findall(r"\b\w+\b", text)
    matches = [w for w in words if w in lexicon and w not in STOPWORDS]
    total_score = sum(lexicon[w] for w in matches)
    return {
        "score": total_score,
        "matches": matches,
        "match_count": len(matches)
    }
