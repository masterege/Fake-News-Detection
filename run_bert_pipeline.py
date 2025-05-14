import os
import json
import torch
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.preprocessing import load_data, preprocess_dataset, split_data
from src.bert_model import BERTNewsClassifier, NewsDataset

# Utilities

def summarize_results(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True)
    }

def save_metrics(metrics_dict, model_name, output_dir="outputs"):
    Path(output_dir).mkdir(exist_ok=True)
    filepath = Path(output_dir) / f"{model_name}_metrics.json"
    with open(filepath, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Saved {model_name} metrics to {filepath}")

# Step 1: Load dataset
base_dir = Path(__file__).resolve().parent
data_path = base_dir / "data" / "fake_news_dataset.csv"
print(f"Loading dataset from: {data_path}")
dataset = load_data(data_path)

# Step 2: Preprocess
dataset = preprocess_dataset(dataset)

# Step 3: Train/test split
X_train, X_test, y_train, y_test = split_data(dataset)

# Step 4: Initialize BERT model
bert_classifier = BERTNewsClassifier()

# Step 5: Train
bert_classifier.train(X_train.tolist(), y_train.tolist(), epochs=2, batch_size=8)

# Step 6: Evaluate
y_pred = []
y_proba = []
y_true = y_test.tolist()

bert_classifier.model.eval()
dataset_test = NewsDataset(X_test.tolist(), y_true, bert_classifier.tokenizer)
dataloader = DataLoader(dataset_test, batch_size=8)

with torch.no_grad():
    for batch in dataloader:
        input_ids = batch['input_ids'].to(bert_classifier.device)
        attention_mask = batch['attention_mask'].to(bert_classifier.device)
        labels = batch['labels'].to(bert_classifier.device)

        outputs = bert_classifier.model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        y_pred.extend(preds.cpu().numpy())
        y_proba.extend(probs[:, 1].cpu().numpy())

# Step 7: Print and save metrics
bert_classifier.evaluate(X_test.tolist(), y_test.tolist(), batch_size=8)

metrics = summarize_results(y_true, y_pred, y_proba)
save_metrics(metrics, model_name="bert")
