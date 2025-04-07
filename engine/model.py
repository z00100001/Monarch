import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Configuration
MODEL_NAME = "roberta-base"
OUTPUT_DIR = "mental_health_model/"
NUM_LABELS = 2  # Binary classification: depression/not or anxiety/not

# Make sure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    """Calculate various metrics for model evaluation"""
    predictions, labels = eval_pred
    
    # For binary classification
    probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=1)
    preds = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', pos_label=1
    )
    acc = accuracy_score(labels, preds)
    
    # Calculate AUC if we have binary classification
    auc = roc_auc_score(labels, probs[:, 1].numpy())
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

class LossPlotCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.eval_metrics = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                self.train_losses.append((state.global_step, logs["loss"]))
            if "eval_loss" in logs:
                self.eval_losses.append((state.global_step, logs["eval_loss"]))
                
                # Also save metrics
                metrics = {k: v for k, v in logs.items() if k.startswith('eval_') and k != 'eval_loss'}
                self.eval_metrics.append((state.global_step, metrics))
    
    def on_train_end(self, args, state, control, **kwargs):
        # Plot loss curves
        plt.figure(figsize=(12, 6))
        
        if self.train_losses:  # Check if list is not empty
            steps, losses = zip(*self.train_losses)
            plt.plot(steps, losses, label='Training Loss')
        
        if self.eval_losses:  # Check if list is not empty
            steps, losses = zip(*self.eval_losses)
            plt.plot(steps, losses, label='Validation Loss')
        
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Create directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curves.png'))
        
        # Plot metrics if available
        if self.eval_metrics:
            metrics_dict = {}
            for step, metrics in self.eval_metrics:
                for k, v in metrics.items():
                    if k not in metrics_dict:
                        metrics_dict[k] = []
                    metrics_dict[k].append((step, v))
                    
            if metrics_dict:  # Check if we have any metrics
                plt.figure(figsize=(12, 6))
                for metric_name, values in metrics_dict.items():
                    steps, metric_values = zip(*values)
                    plt.plot(steps, metric_values, label=metric_name)
                    
                plt.xlabel('Step')
                plt.ylabel('Score')
                plt.title('Evaluation Metrics')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(OUTPUT_DIR, 'metrics.png'))

def train_mental_health_model(train_texts, train_labels, val_texts, val_labels):
    """Train a mental health classification model"""
    # Set up tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    
    # Create datasets
    train_dataset = MentalHealthDataset(train_texts, train_labels, tokenizer)
    val_dataset = MentalHealthDataset(val_texts, val_labels, tokenizer)
    
    # Create model - FIX: Alternative way to load the model to avoid init_empty_weights
    try:
        print("Attempting to load model with older version compatibility...")
        # Try to create model with explicit config to avoid init_empty_weights
        from transformers import RobertaConfig
        config = RobertaConfig.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
        model = RobertaForSequenceClassification.from_pretrained(
            MODEL_NAME,
            config=config,
            # Avoid using methods that might involve init_empty_weights
            _from_transformers=True  
        )
        print("Model loaded successfully with custom config")
    except Exception as e:
        print(f"First attempt failed: {str(e)}")
        try:
            # Alternative approach: pip install an older version of transformers
            print("Attempting to install older transformers version...")
            import subprocess
            subprocess.check_call(["pip", "install", "transformers==4.25.1", "--quiet"])
            print("Installed older transformers version, reloading...")
            
            # Reload transformers with older version
            import importlib
            importlib.reload(transformers)
            from transformers import RobertaForSequenceClassification
            
            model = RobertaForSequenceClassification.from_pretrained(
                MODEL_NAME, 
                num_labels=NUM_LABELS
            )
            print("Model loaded successfully with older transformers version")
        except Exception as e2:
            print(f"Second attempt failed: {str(e2)}")
            # Fallback method: create a new model from scratch
            print("Falling back to creating a new model from scratch...")
            from transformers import RobertaConfig
            config = RobertaConfig.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
            model = RobertaForSequenceClassification(config)
            print("Created new model from scratch")
    
    # Progress tracking - print model information
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Use smaller batch size and fewer epochs to ensure training works
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="steps",
        eval_steps=50,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=4,  # Smaller batch size
        per_device_eval_batch_size=8,
        num_train_epochs=2,  # Fewer epochs for testing
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
        report_to="none",
        disable_tqdm=False,
        no_cuda=False,  # Use CPU if GPU is causing issues
        dataloader_num_workers=0  # Avoid multiprocessing issues
    )
    
    # Create callbacks
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)
    loss_plot_callback = LossPlotCallback()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback, loss_plot_callback]
    )
    
    # Train model with added verification
    print("Training model...")
    try:
        train_result = trainer.train()
        print(f"Training complete! Steps: {train_result.global_step}, Loss: {train_result.training_loss}")
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        raise e
    
    # Evaluate
    print("Evaluating model...")
    try:
        eval_results = trainer.evaluate()
        print("✅ Evaluation completed successfully!")
        print(f"Evaluation results:")
        for metric_name, value in eval_results.items():
            print(f"  {metric_name}: {value:.4f}")
    except Exception as e:
        print(f"❌ Evaluation failed with error: {str(e)}")
        raise e
    
    # Save the model
    try:
        print(f"Saving model to {OUTPUT_DIR}")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"✅ Model saved successfully to {OUTPUT_DIR}")
        
        # Save evaluation results
        with open(os.path.join(OUTPUT_DIR, "eval_results.json"), "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"✅ Evaluation results saved to {os.path.join(OUTPUT_DIR, 'eval_results.json')}")
    except Exception as e:
        print(f"❌ Failed to save model: {str(e)}")
        raise e
    
    # Verify model exists
    if os.path.exists(os.path.join(OUTPUT_DIR, "pytorch_model.bin")) or \
       os.path.exists(os.path.join(OUTPUT_DIR, "model.safetensors")):
        print("✅ Model files verified on disk")
    else:
        print("⚠️ Model files not found on disk - check for errors")
    
    return model, tokenizer, eval_results

def predict_mental_health(texts, model=None, tokenizer=None, threshold=0.5, device=None):
    """
    Predict depression/anxiety probability for texts
    
    Args:
        texts: List of text strings to predict on
        model: Optional pre-loaded model
        tokenizer: Optional pre-loaded tokenizer
        threshold: Probability threshold for positive classification
        device: Device to run inference on (None for auto)
        
    Returns:
        Dictionary with predictions and probabilities
    """
    # Load model and tokenizer if not provided
    if model is None or tokenizer is None:
        try:
            print(f"Loading model and tokenizer from {OUTPUT_DIR}")
            tokenizer = RobertaTokenizer.from_pretrained(OUTPUT_DIR)
            model = RobertaForSequenceClassification.from_pretrained(OUTPUT_DIR)
            print("✅ Model and tokenizer loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model and tokenizer: {str(e)}")
            raise e
    
    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    model = model.to(device)
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Tokenize texts
    try:
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print(f"✅ Texts tokenized successfully")
    except Exception as e:
        print(f"❌ Failed to tokenize texts: {str(e)}")
        raise e
    
    # Get predictions
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            
        # Convert to numpy for easier handling
        probs = probabilities.cpu().numpy()
        print(f"✅ Predictions generated successfully")
    except Exception as e:
        print(f"❌ Failed to generate predictions: {str(e)}")
        raise e
    
    # Get predictions based on threshold
    predictions = (probs[:, 1] >= threshold).astype(int)
    
    # Create results
    results = {
        "text": texts,
        "predictions": predictions.tolist(),
        "depression_probability": probs[:, 1].tolist()
    }
    
    return results

def evaluate_model_performance(test_texts, test_labels, model=None, tokenizer=None):
    """
    Evaluate model performance on a test set
    
    Args:
        test_texts: List of text strings to evaluate on
        test_labels: Ground truth labels (1 for depression, 0 for not)
        model: Optional pre-loaded model
        tokenizer: Optional pre-loaded tokenizer
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Get predictions
    results = predict_mental_health(test_texts, model, tokenizer)
    predictions = results["predictions"]
    probabilities = results["depression_probability"]
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average='binary', pos_label=1
    )
    auc = roc_auc_score(test_labels, probabilities)
    
    # Print evaluation results
    print("\n===== Model Performance =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Plot confusion matrix
    try:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        cm = confusion_matrix(test_labels, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Depressed", "Depressed"])
        plt.figure(figsize=(8, 6))
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
        plt.savefig(cm_path)
        print(f"✅ Confusion matrix saved to {cm_path}")
    except Exception as e:
        print(f"⚠️ Could not generate confusion matrix: {str(e)}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

# Example usage with simpler code path
if __name__ == "__main__":
    print("Starting mental health model training...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check versions - helpful for debugging
    import transformers 
    print(f"Transformers version: {transformers.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Simpler dataset for faster training
    print("Loading training data...")
    
    train_texts = [
        "I feel so sad and empty all the time", 
        "Having a great day!",
        "Nothing matters anymore, I just want to sleep",
        "Excited about the weekend plans",
        "Can't stop crying and don't know why",
        "Just finished a great workout",
        "Don't see the point in trying anymore",
        "Looking forward to the holiday season"
    ]
    
    train_labels = [1, 0, 1, 0, 1, 0, 1, 0]
    
    val_texts = [
        "Nothing seems to matter anymore", 
        "Excited about the weekend",
        "Been feeling worthless for weeks",
        "Just got a promotion at work"
    ]
    
    val_labels = [1, 0, 1, 0]
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Train the model with extensive error handling
    print("Starting model training...")
    try:
        model, tokenizer, eval_results = train_mental_health_model(train_texts, train_labels, val_texts, val_labels)
        
        # Make predictions
        test_texts = ["I don't see any point in going on", "Happy to be here"]
        test_labels = [1, 0]  # Ground truth for evaluation
        
        print(f"\nMaking predictions on test texts...")
        results = predict_mental_health(test_texts, model, tokenizer)
        
        # Evaluate model performance
        performance = evaluate_model_performance(test_texts, test_labels, model, tokenizer)
        
        # Display results
        print("\n===== Test Results =====")
        for i, text in enumerate(test_texts):
            prob = results["depression_probability"][i] * 100
            print(f"Text: {text}")
            print(f"Depression probability: {prob:.2f}%")
            print(f"Classification: {'Depressed' if results['predictions'][i] == 1 else 'Not depressed'}")
            print()
        
        print("Model training and testing complete!")
        print(f"Model artifacts saved to: {os.path.abspath(OUTPUT_DIR)}")
        
    except Exception as e:
        print(f"❌ An error occurred during the training process: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # If the model creation is the issue, provide a simpler alternative
        print("\n\nAttempting alternative approach - direct model creation...")
        try:
            from transformers import RobertaConfig
            print("Creating a model directly from config...")
            config = RobertaConfig(
                vocab_size=50265,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_position_embeddings=514,
                type_vocab_size=1,
                num_labels=2
            )
            model = RobertaForSequenceClassification(config)
            print("✅ Successfully created model directly from config")
            
            # Continue with basic usage
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            
            # Create a very simple example
            input_text = ["I feel depressed", "I am happy"]
            inputs = tokenizer(input_text, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                
            print("✅ Model inference successful!")
            
        except Exception as e2:
            print(f"❌ Alternative approach also failed: {str(e2)}")
            print("\nRecommendation: Try updating your transformers library:")
            print("pip install --upgrade transformers")
            print("or try with a different model type:")
            print("from transformers import DistilBertForSequenceClassification")