import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Dynamically resolve the current file's directory
base_dir = Path(__file__).resolve().parent
output_dir = base_dir / "outputs"
output_dir.mkdir(exist_ok=True)

# Define file paths
tfidf_file = output_dir / "tfidf_xgb_metrics.json"
bert_file = output_dir / "bert_metrics.json"

# Load JSON files
def load_metrics(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

metrics_tfidf = load_metrics(tfidf_file)
metrics_bert = load_metrics(bert_file)

# Prepare data for plotting
models = ['TF-IDF + XGBoost', 'BERT']
accuracy = [metrics_tfidf['accuracy'], metrics_bert['accuracy']]
f1_score = [metrics_tfidf['f1_score'], metrics_bert['f1_score']]
roc_auc = [metrics_tfidf['roc_auc'], metrics_bert['roc_auc']]

# Plot and save comparison charts
def plot_metric(values, title, ylabel, filename):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=models, y=values, palette='Set2')
    plt.ylim(0, 1)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()

    # Save plot to file
    filepath = output_dir / filename
    plt.savefig(filepath)
    print(f"Saved plot: {filepath}")
    plt.close()

plot_metric(accuracy, 'Accuracy Comparison', 'Accuracy', 'accuracy_comparison.png')
plot_metric(f1_score, 'F1 Score Comparison', 'F1 Score', 'f1_comparison.png')
plot_metric(roc_auc, 'ROC AUC Comparison', 'ROC AUC', 'roc_auc_comparison.png')