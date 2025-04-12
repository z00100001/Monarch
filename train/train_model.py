import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from datasets import Dataset
import torch
from tqdm.auto import tqdm

GOEMOTIONS_PATH = "data/processed/goemotions_cleaned.json"
POSITIVE_PATH = "data/processed/positive/reddit_positive_scored.json"
NEGATIVE_PATH = "data/processed/reddit_scored.json"
MODEL_OUTPUT_DIR = "model/"

# labels
emotion_map = {
    "sadness": 0,
    "anger": 1,
    "fear": 2,
    "joy": 3,
    "neutral": 4,
    "worry": 5
}

# load and pre-process
def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
        texts = []
        labels = []
    for entry in data:
        label = entry.get("labels", ["neutral"])[0]
        if label in emotion_map:
            text = entry.get("clean_text", entry.get("text", ""))
            texts.append(text)
            labels.append(emotion_map[label])

    return texts, labels


def tokenize_data(texts, labels, tokenizer):
    encodings = tokenizer(texts, truncation=True, padding=True)
    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels
    })
    return dataset

# progress bar
class TQDMProgressBarCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.progress_bar = tqdm(total=state.max_steps, desc="🔥 Training Progress")
    def on_step_end(self, args, state, control, **kwargs):
        self.progress_bar.update(1)
    def on_train_end(self, args, state, control, **kwargs):
        self.progress_bar.close()


class LossTrackerCallback(TrainerCallback):
    def __init__(self):
        self.train_loss = []
        self.eval_loss = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.train_loss.append(logs["loss"])
        if "eval_loss" in logs:
            self.eval_loss.append(logs["eval_loss"])

    def on_train_end(self, args, state, control, **kwargs):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_loss, label="Training Loss", marker="o")
        if self.eval_loss:
            plt.plot(self.eval_loss, label="Validation Loss", marker="x")
        plt.title("Loss Over Epochs")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("loss_plot.png")
        plt.show()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

def train_on_dataset(path, model, tokenizer, args, stage):
    print(f"\nLoading {stage} data...")
    texts, labels = load_data(path)
    dataset = tokenize_data(texts, labels, tokenizer)
    train_ds, eval_ds = dataset.train_test_split(test_size=0.2).values()

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[TQDMProgressBarCallback(), LossTrackerCallback()]
    )

    print(f"Training on {stage} data...")
    trainer.train()

    return model

# --- Main function ---
def main():
    print("Starting Monarch multi-source training...")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)

    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=False
    )

    model = train_on_dataset(GOEMOTIONS_PATH, model, tokenizer, training_args, "GoEmotions")
    model = train_on_dataset(POSITIVE_PATH, model, tokenizer, training_args, "Positive Reddit")
    model = train_on_dataset(NEGATIVE_PATH, model, tokenizer, training_args, "Negative Reddit")

    print(f"\nSaving model to {MODEL_OUTPUT_DIR}")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

if __name__ == "__main__":
    main()