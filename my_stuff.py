import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import argparse
import os

# ==========================
# 1. Data Preparation
# ==========================

def load_data(file_path):
    """
    Load the dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    data = pd.read_csv(file_path)
    if 'policy_title' not in data.columns or 'policy_summary' not in data.columns:
        raise ValueError("CSV must contain 'policy_title' and 'policy_summary' columns.")
    # Combine policy title and summary into a single text field
    data['text'] = data['policy_title'].astype(str) + " " + data['policy_summary'].astype(str)
    return data

def encode_labels(data, num_departments=12):
    """
    Encode department labels from the DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing department labels.
        num_departments (int): Number of departments.

    Returns:
        np.ndarray: Encoded labels.
    """
    label_cols = [f'dept_{i}' for i in range(1, num_departments + 1)]
    for col in label_cols:
        if col not in data.columns:
            raise ValueError(f"Missing column: {col} in the dataset.")
    labels = data[label_cols].values
    return labels, label_cols

def split_data(texts, labels, test_size=0.2, random_state=42):
    """
    Split the data into training and validation sets.

    Args:
        texts (list): List of text data.
        labels (np.ndarray): Array of labels.
        test_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed.

    Returns:
        Tuple: Train and validation texts and labels.
    """
    return train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state
    )

# ==========================
# 2. BERT Embedding Extraction
# ==========================

class PolicyDataset(Dataset):
    """
    Custom Dataset for Policy data.
    """
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()  # Shape: (max_length)
        attention_mask = encoding['attention_mask'].squeeze()  # Shape: (max_length)
        labels = torch.FloatTensor(self.labels[idx])  # Shape: (num_departments)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# ==========================
# 3. Neural Network Construction
# ==========================

class PolicyClassifier(nn.Module):
    """
    Neural Network Model for Policy Classification.
    """
    def __init__(self, bert_model, dropout=0.3, num_classes=12):
        super(PolicyClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    cls_output = outputs.last_hidden_state[:, 0, :]
    x = self.dropout(cls_output)
    logits = self.classifier(x)
    return logits

    def forward(self, input_ids, attention_mask):
        # If you want to fine-tune BERT, remove the following line
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        x = self.dropout(cls_output)
        logits = self.classifier(x)
        return logits





# ==========================
# 4. Training and Evaluation Functions
# ==========================

def train_epoch(model, data_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        data_loader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to run the training on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    epoch_loss = 0

    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(data_loader)

def eval_model(model, data_loader, criterion, device):
    """
    Evaluate the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the evaluation on.

    Returns:
        Tuple: Average loss, accuracy, precision, recall, and F1-score.
    """
    model.eval()
    epoch_loss = 0
    preds = []
    true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            preds.append(torch.sigmoid(outputs).cpu().numpy())
            true_labels.append(labels.cpu().numpy())

    preds = np.vstack(preds)
    true_labels = np.vstack(true_labels)
    preds_binary = (preds >= 0.5).astype(int)

    accuracy = accuracy_score(true_labels, preds_binary)
    precision = precision_score(true_labels, preds_binary, average='micro', zero_division=0)
    recall = recall_score(true_labels, preds_binary, average='micro', zero_division=0)
    f1 = f1_score(true_labels, preds_binary, average='micro', zero_division=0)

    return epoch_loss / len(data_loader), accuracy, precision, recall, f1

# ==========================
# 5. Inference Function
# ==========================

def predict(model, tokenizer, text, device, threshold=0.5, max_length=512):
    """
    Make a prediction on a single text input.

    Args:
        model (nn.Module): The trained model.
        tokenizer (BertTokenizer): BERT tokenizer.
        text (str): The input text.
        device (torch.device): Device to run the inference on.
        threshold (float): Threshold for classification.
        max_length (int): Maximum sequence length.

    Returns:
        Tuple: Binary predictions and probabilities for each class.
    """
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    preds = (probs >= threshold).astype(int)
    return preds, probs

# ==========================
# 6. Main Training Loop
# ==========================

def main(args):
    # Check device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    data = load_data(args.data_path)
    labels, label_cols = encode_labels(data, num_departments=12)
    train_texts, val_texts, train_labels, val_labels = split_data(
        data['text'].tolist(),
        labels,
        test_size=0.2,
        random_state=42
    )

    # Initialize tokenizer and BERT model
    print("Initializing BERT tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.to(device)
    bert_model.eval()  # Set BERT to evaluation mode

    # Create datasets and dataloaders
    print("Creating datasets and dataloaders...")
    train_dataset = PolicyDataset(train_texts, train_labels, tokenizer, max_length=512)
    val_dataset = PolicyDataset(val_texts, val_labels, tokenizer, max_length=512)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Initialize the model
    print("Initializing the classifier model...")
    model = PolicyClassifier(bert_model, dropout=0.3, num_classes=12)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    print("Starting training...")
    best_f1 = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = eval_model(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f}")

        # Save the model if F1 improves
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), args.save_path)
            print(f"Model saved to {args.save_path} (F1 improved to {best_f1:.4f})")

    print("\nTraining complete!")

    # Example inference (optional)
    if args.example_text:
        print("\nPerforming example prediction...")
        preds, probs = predict(model, tokenizer, args.example_text, device)
        for idx, (pred, prob) in enumerate(zip(preds, probs), start=1):
            print(f"Department {idx}: {'Applicable' if pred else 'Not Applicable'} (Probability: {prob:.4f})")

# ==========================
# 7. Entry Point
# ==========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT-based Multi-label Policy Classifier")

    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to the CSV file containing the dataset.'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='policy_classifier.pt',
        help='Path to save the trained model.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for training and evaluation.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=2e-5,
        help='Learning rate for the optimizer.'
    )
    parser.add_argument(
        '--example_text',
        type=str,
        default=None,
        help='Example text for making a prediction after training.'
    )

    args = parser.parse_args()
    main(args)
