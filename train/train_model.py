import json
import os
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset
import torch
from tqdm.auto import tqdm

GOEMOTIONS_PATH = "data/processed/goemotions_cleaned.json"
MODEL_OUTPUT_DIR = "model/"

# Load and preprocess data
def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)

    texts = []
    labels = []

    emotion_map = {"sadness": 0, "anger": 1, "fear": 2, "joy": 3, "neutral": 4}
    
    for entry in data:
        label = entry["labels"][0] if entry["labels"] else "neutral"
        if label in emotion_map:
            texts.append(entry["clean_text"])
            labels.append(emotion_map[label])

    return texts, labels

# tokenizer
def tokenize_data(texts, labels, tokenizer):
    encodings = tokenizer(texts, truncation=True, padding=True)
    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels
    })
    return dataset

class TQDMProgressBarCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.progress_bar = tqdm(total=state.max_steps, desc="🔥 Training Progress")

    def on_step_end(self, args, state, control, **kwargs):
        self.progress_bar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        self.progress_bar.close()

def main():
    print("Loading data...")
    texts, labels = load_data(GOEMOTIONS_PATH)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset = tokenize_data(texts, labels, tokenizer)

    print("Splitting data...")
    train_ds, eval_ds = dataset.train_test_split(test_size=0.2).values()

    print("Initializing model...")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)  # Ensure directory exists

    args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=[TQDMProgressBarCallback()]
    )

    print("Training started...")
    trainer.train()

    print(f"Saving model to {MODEL_OUTPUT_DIR}")
    try:
        trainer.save_model(MODEL_OUTPUT_DIR)
    except Exception as e:
        print(f"Failed to save model with Trainer. Falling back to manual save. Error: {e}")
        model.save_pretrained(MODEL_OUTPUT_DIR)
        tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

if __name__ == "__main__":
    main()
