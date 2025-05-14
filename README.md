# Fake News Detection Project 📰

This project demonstrates a **machine learning pipeline** to detect fake news using:
- **TF-IDF + XGBoost**
- **BERT (Transformer-based model)**

It includes data preprocessing, model training, evaluation, and comparison visualizations.


## 🚀 How to Run

### 1. ⚙️ Install Dependencies
```bash
pip install -r requirements.txt
```
> Make sure you have `transformers`, `torch`, `xgboost`, `scikit-learn`, `matplotlib`, `seaborn`, `nltk`

### 2. ▶️ Run Full Pipeline
```bash
python main.py
```
This will:
- Train both models
- Save their metrics to JSON
- Generate comparison plots in `outputs/`

---

## 📊 Features

### ✅ Preprocessing
- Lowercasing, punctuation/number removal
- Tokenization, stopword removal
- Lemmatization
- Feature creation: `title_length`, `text_length`, `clean_text`

### ✅ Models
- **TF-IDF + XGBoost** with `RandomizedSearchCV`
- **BERT** fine-tuned with `transformers`

### ✅ Evaluation
- Accuracy, F1 Score, ROC AUC
- Confusion Matrix
- Bar plots comparing models

---

## 🧪 Example Metrics (Sample Output)
| Model              | Accuracy | F1 Score | ROC AUC |
|-------------------|----------|----------|---------|
| TF-IDF + XGBoost  | 0.50     | 0.51     | 0.51    |
| BERT              | 0.68     | 0.69     | 0.75    |

> 📉 If metrics look random (~0.5), inspect class imbalance or training loop.

---

## 📌 Notes
- The dataset is **synthetic**, meant for practice only.
- The results are indicating the model is almost randomly guessing(Around 0.50). Due to limited computational power; number of epochs, and batch size values are low, while learning rate is high.
- Change values in the "bert_model.py" file in the "src" file for better results.
---

## 📬 Contact
Created by **Ege Ebiller** — feel free to fork, modify, or reach out with improvements!
