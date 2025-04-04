import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer

with open("data/processed/goemotions_train.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

df = df[df['labels'].apply(lambda x: isinstance(x, list) and len(x) > 0)]

# --- Plot 1: Emotion Frequency ---
all_labels = [label for sublist in df['labels'] for label in sublist]
label_counts = Counter(all_labels)

plt.figure(figsize=(12, 6))
sns.barplot(x=list(label_counts.keys()), y=list(label_counts.values()))
plt.xticks(rotation=45)
plt.title("GoEmotions: Emotion Frequency")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("reports/emotion_frequency.png")
plt.show()

# --- Plot 2: Emotion Co-occurrence Heatmap ---
mlb = MultiLabelBinarizer()
binary_matrix = mlb.fit_transform(df["labels"])
binary_df = pd.DataFrame(binary_matrix, columns=mlb.classes_)

correlation_matrix = binary_df.corr()

plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, cmap="coolwarm", center=0, annot=False)
plt.title("GoEmotions: Emotion Co-occurrence Correlation")
plt.tight_layout()
plt.savefig("reports/emotion_correlation.png")
plt.show()
