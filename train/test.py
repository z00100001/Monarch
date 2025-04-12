from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
import torch.nn.functional as F

model_dir = "model"
tokenizer = BertTokenizerFast.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)
model.eval()

text = input("Enter Paragraph: ")

inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=1)
    predicted_class_id = torch.argmax(probs, dim=1).item()

id2label = {
    0: "sadness",
    1: "anger",
    2: "distress",
    3: "joy",
    4: "worry"
}

label = id2label.get(predicted_class_id, f"Label {predicted_class_id}")
print("sadness: 0, anger: 1, distress: 2, joy: 3, worry: 4")
print(f"Input: {text}")
print(f"Prediction: {label} (class {predicted_class_id})")
print(f"🔥Confidence: {probs[0][predicted_class_id].item():.4f}")

print("\nEmotion Probabilities:")
for i, score in enumerate(probs[0]):
    print(f"{id2label[i]:<10}: {score.item():.4f}")
