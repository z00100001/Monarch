from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
import torch.nn.functional as F

model_dir = "model"
tokenizer = BertTokenizerFast.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)
model.eval()

id2label = {
    0: "sadness",
    1: "anger",
    2: "distress",
    3: "joy",
    4: "worry"
}

print("Emotion Index: sadness: 0, anger: 1, distress: 2, joy: 3, worry: 4")

while True:
    text = input("\nEnter Paragraph (or type 'exit'): ").strip()
    if text.lower() in {"exit", "quit"}:
        break

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)[0] 

    predicted_class_id = torch.argmax(probs).item()
    top_emotion = id2label[predicted_class_id]
    confidence = probs[predicted_class_id].item()

    print(f"\n🔥 Predicted Dominant Emotion: {top_emotion} (class {predicted_class_id})")
    print(f"🔥 Confidence: {confidence:.4f}")

    print("Emotion Distribution:")
    for i, score in enumerate(probs):
        bar = "█" * int(score.item() * 40)
        print(f"{id2label[i]:<10}: {score.item():.4f}  {bar}")
