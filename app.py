from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

model_path = "models/saved_bert_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

@app.route("/classify", methods=["POST"])
def classify_resume():
    resume_text = request.json.get("text", "")
    inputs = tokenizer(resume_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()

    categories = [
    "HR", "Designer", "Information-Technology", "Teacher", "Advocate",
    "Business-Development", "Healthcare", "Fitness", "Agriculture", "BPO",
    "Sales", "Consultant", "Digital-Media", "Automobile", "Chef", "Finance",
    "Apparel", "Engineering", "Accountant", "Construction", "Public-Relations",
    "Banking", "Arts", "Aviation"
]
  # List all categories here
    category = categories[predicted_class_id]
    return jsonify({"category": category})

if __name__ == "__main__":
    app.run(debug=True)
