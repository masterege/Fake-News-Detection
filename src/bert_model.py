import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from tqdm import tqdm

# Detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class BERTNewsClassifier:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.model.to(device)
        self.device = device

    def train(self, train_texts, train_labels, epochs=4, batch_size=8, lr=2e-5):
        train_dataset = NewsDataset(train_texts, train_labels, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=lr)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)
            for batch in loop:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            accuracy = correct / total
            print(f"Epoch {epoch + 1} - Loss: {total_loss:.4f} | Training Accuracy: {accuracy:.4f}")

    def evaluate(self, test_texts, test_labels, batch_size=16):
        test_dataset = NewsDataset(test_texts, test_labels, self.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        self.model.eval()

        all_preds, all_probs, all_labels = [], [], []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print("\nClassification Report:\n", classification_report(all_labels, all_preds))
        print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
        print("ROC AUC Score:", roc_auc_score(all_labels, all_probs))
