import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def load_data(filepath):
    """Loads the dataset from a CSV file."""
    dataset = pd.read_csv(filepath)
    return dataset

def clean_text(text):
    """Cleans and preprocesses raw text."""
    if pd.isnull(text):
        return ""

    text = text.lower()
    text = re.sub(f"[{string.punctuation}0-9]", " ", text)
    tokens = nltk.word_tokenize(text)
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(cleaned)

def preprocess_dataset(dataset):
    """Cleans and adds engineered features to the dataset."""
    dataset['title_length'] = dataset['title'].astype(str).apply(len)
    dataset['text_length'] = dataset['text'].astype(str).apply(len)
    dataset['full_text'] = dataset['title'].astype(str) + " " + dataset['text'].astype(str)
    dataset['clean_text'] = dataset['full_text'].apply(clean_text)
    dataset['label'] = dataset['label'].map({'real': 0, 'fake': 1})  # Ensure binary
    return dataset

def split_data(dataset, test_size=0.2, random_state=42):
    """Splits the dataset into train and test sets."""
    X = dataset['clean_text']
    y = dataset['label']
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)