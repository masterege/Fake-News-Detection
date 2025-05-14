import os
import json
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier

from src.preprocessing import load_data, preprocess_dataset

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
dataset = preprocess_dataset(dataset)

# Step 2: Split dataset
X_train_text, X_test_text, y_train, y_test = train_test_split(
    dataset['clean_text'], dataset['label'], test_size=0.2, random_state=42, stratify=dataset['label']
)

# Step 3: TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = tfidf_vectorizer.fit_transform(X_train_text)
X_test = tfidf_vectorizer.transform(X_test_text)

# Step 4: Hyperparameter tuning with RandomizedSearchCV
print("\nTuning hyperparameters with RandomizedSearchCV...")
param_dist = {
    'n_estimators': [100, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8],
    'gamma': [0, 1, 5]
}

base_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
search = RandomizedSearchCV(
    base_model,
    param_distributions=param_dist,
    n_iter=5,
    scoring='roc_auc',
    cv=2,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)
best_model = search.best_estimator_
print("Best parameters:", search.best_params_)

# Step 5: Predict and evaluate
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# Step 6: Save metrics
metrics = summarize_results(y_test, y_pred, y_proba)
save_metrics(metrics, model_name="tfidf_xgb")