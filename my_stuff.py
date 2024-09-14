# Install required packages (uncomment and run if needed)
# !pip install transformers sklearn pandas numpy torch

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score
)
from collections import Counter
import os
import copy
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================
# 1. Data Loading and Preparation
# ==========================

# Load your dataset
# Replace 'policies.csv' with your actual data file path
data = pd.read_csv('policies.csv')

# Ensure that 'policy_title' and 'policy_summary' are strings
data['policy_title'] = data['policy_title'].astype(str)
data['policy_summary'] = data['policy_summary'].astype(str)

# Combine 'policy_title' and 'policy_summary' into one text column
data['text'] = data['policy_title'] + ' ' + data['policy_summary']

# Extract department labels (assuming columns are named 'dept_1' to 'dept_12')
num_departments = 12
label_cols = [f'dept_{i}' for i in range(1, num_departments + 1)]

# Check if all label columns exist
for col in label_cols:
    if col not in data.columns:
        raise ValueError(f"Column {col} not found in the dataset.")

# Extract labels
labels = data[label_cols].values

# Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['text'].tolist(),
    labels,
    test_size=0.2,
    random_state=42
)

# ==========================
# 2. Handling Imbalanced Data
# ==========================

# Compute initial class weights based on training data
def compute_class_weights(labels):
    # labels: numpy array of shape (num_samples, num_classes)
    class_weights = []
    for i in range(labels.shape[1]):
        class_counts = Counter(labels[:, i])
        neg_count = class_counts.get(0, 0)
        pos_count = class_counts.get(1, 0)
        # Avoid division by zero
        if pos_count == 0:
            pos_count = 1
        weight = neg_count / pos_count
        class_weights.append(weight)
    return torch.FloatTensor(class_weights)

initial_class_weights = compute_class_weights(train_labels)
print("Initial Class Weights:", initial_class_weights)

# ==========================
# 3. Dataset and DataLoader
# ==========================

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define custom dataset
class PolicyDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
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
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),  # Shape: (max_length)
            'attention_mask': encoding['attention_mask'].flatten(),  # Shape: (max_length)
        }
        
        if self.labels is not None:
            item['labels'] = torch.FloatTensor(self.labels[idx])  # Shape: (num_departments)
        
        return item

# Create datasets
train_dataset = PolicyDataset(train_texts, train_labels, tokenizer)
val_dataset = PolicyDataset(val_texts, val_labels, tokenizer)

# Create dataloaders
batch_size = 8  # Adjust based on your GPU memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ==========================
# 4. Model Definition
# ==========================

# Define the classifier model
class PolicyClassifier(nn.Module):
    def __init__(self, num_classes=12, dropout=0.3):
        super(PolicyClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # Get the outputs from BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Extract the [CLS] token embeddings
        cls_output = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
        x = self.dropout(cls_output)
        logits = self.classifier(x)  # Shape: (batch_size, num_classes)
        return logits

# Function to train and evaluate the model
def train_and_evaluate(model, criterion, optimizer, train_loader, val_loader, device, epochs=3):
    best_model = None
    best_macro_f1 = 0
    last_epoch_metrics = None  # To store metrics of the last epoch
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.sigmoid(outputs)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
        avg_val_loss = val_loss / len(val_loader)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        preds_binary = (all_preds >= 0.5).int()
        
        # Calculate overall macro F1-score
        macro_f1 = f1_score(
            all_labels.numpy(), preds_binary.numpy(), average='macro', zero_division=0
        )
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Macro F1 Score: {macro_f1:.4f}")
        print("-" * 50)
        
        # Save the best model based on macro F1-score
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_model = copy.deepcopy(model.state_dict())
            # Store metrics of the last epoch
            last_epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'macro_f1': macro_f1,
                'all_labels': all_labels.numpy(),
                'preds_binary': preds_binary.numpy()
            }
    
    return best_model, best_macro_f1, last_epoch_metrics

# ==========================
# 5. Grid Search over Scaling Factors
# ==========================

# Define scaling factors to apply to the initial class weights
scaling_factors = [0.5, 1.0, 2.0, 5.0]

# Store results for each scaling factor
grid_search_results = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for scale in scaling_factors:
    print(f"Grid Search - Scaling Factor: {scale}")
    # Scale the class weights
    scaled_class_weights = initial_class_weights * scale
    
    # Define loss function with scaled class weights
    criterion = nn.BCEWithLogitsLoss(pos_weight=scaled_class_weights.to(device))
    
    # Initialize the model
    model = PolicyClassifier(num_classes=num_departments)
    model = model.to(device)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    
    # Train and evaluate the model
    start_time = time.time()
    best_model_state, best_macro_f1, last_epoch_metrics = train_and_evaluate(
        model, criterion, optimizer, train_loader, val_loader, device, epochs=3
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Store the results
    grid_search_results.append({
        'scaling_factor': scale,
        'best_macro_f1': best_macro_f1,
        'training_time': elapsed_time,
        'best_model_state': best_model_state,  # Save the best model state
        'last_epoch_metrics': last_epoch_metrics  # Save metrics from the last epoch
    })
    print(f"Scaling Factor {scale} - Best Macro F1: {best_macro_f1:.4f}")
    print(f"Training Time: {elapsed_time/60:.2f} minutes")
    print("=" * 60)

# Find the scaling factor with the best macro F1-score
best_result = max(grid_search_results, key=lambda x: x['best_macro_f1'])
optimal_scaling_factor = best_result['scaling_factor']
best_macro_f1 = best_result['best_macro_f1']
best_model_state = best_result['best_model_state']
best_last_epoch_metrics = best_result['last_epoch_metrics']

print(f"Optimal Scaling Factor: {optimal_scaling_factor}")
print(f"Best Macro F1 Score: {best_macro_f1:.4f}")

# ==========================
# 6. Saving Validation Metrics of the Best Scaling Factor
# ==========================

# Extract metrics from the last epoch of the best scaling factor
all_labels = best_last_epoch_metrics['all_labels']
preds_binary = best_last_epoch_metrics['preds_binary']
avg_train_loss = best_last_epoch_metrics['train_loss']
avg_val_loss = best_last_epoch_metrics['val_loss']
epoch = best_last_epoch_metrics['epoch']

# Calculate per-department metrics
per_dept_metrics = []
with open('validation_metrics.txt', 'w') as f:
    f.write(f"Best Scaling Factor: {optimal_scaling_factor}\n")
    f.write(f"Epoch {epoch}\n")
    f.write(f"Train Loss: {avg_train_loss:.4f}\n")
    f.write(f"Validation Loss: {avg_val_loss:.4f}\n\n")
    f.write("Per-Department Metrics:\n")
    f.write("Department\tPrecision\tRecall\tF1-Score\tSupport\n")
    
    for i in range(num_departments):
        dept_name = label_cols[i]
        y_true = all_labels[:, i]
        y_pred = preds_binary[:, i]
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        support = int(y_true.sum())
        
        per_dept_metrics.append({
            'department': dept_name,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support
        })
        
        # Write to file
        f.write(f"{dept_name}\t{precision:.4f}\t{recall:.4f}\t{f1:.4f}\t{support}\n")
    
    # Overall Metrics
    micro_precision = precision_score(
        all_labels, preds_binary, average='micro', zero_division=0
    )
    micro_recall = recall_score(
        all_labels, preds_binary, average='micro', zero_division=0
    )
    micro_f1 = f1_score(
        all_labels, preds_binary, average='micro', zero_division=0
    )
    
    macro_precision = precision_score(
        all_labels, preds_binary, average='macro', zero_division=0
    )
    macro_recall = recall_score(
        all_labels, preds_binary, average='macro', zero_division=0
    )
    macro_f1 = f1_score(
        all_labels, preds_binary, average='macro', zero_division=0
    )
    
    f.write("\nOverall Metrics:\n")
    f.write(f"Micro Precision: {micro_precision:.4f}\n")
    f.write(f"Micro Recall: {micro_recall:.4f}\n")
    f.write(f"Micro F1 Score: {micro_f1:.4f}\n")
    f.write(f"Macro Precision: {macro_precision:.4f}\n")
    f.write(f"Macro Recall: {macro_recall:.4f}\n")
    f.write(f"Macro F1 Score: {macro_f1:.4f}\n")

print("Validation metrics of the best scaling factor saved to 'validation_metrics.txt'")

# Load the best model state
final_model = PolicyClassifier(num_classes=num_departments)
final_model.load_state_dict(best_model_state)
final_model = final_model.to(device)
final_model.eval()

# ==========================
# 7. Saving the Final Model and Tokenizer
# ==========================

# Create directory to save model and tokenizer
model_dir = 'policy_classifier_model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the final model
model_save_path = os.path.join(model_dir, 'model.pt')
torch.save(final_model.state_dict(), model_save_path)

# Save tokenizer
tokenizer_save_path = os.path.join(model_dir, 'tokenizer')
tokenizer.save_pretrained(tokenizer_save_path)

print("Final model and tokenizer saved.")

# ==========================
# 8. Inference on New Data
# ==========================

# Load tokenizer
loaded_tokenizer = BertTokenizer.from_pretrained(tokenizer_save_path)

# Load classifier model
loaded_model = PolicyClassifier(num_classes=num_departments)
loaded_model.load_state_dict(torch.load(model_save_path))
loaded_model = loaded_model.to(device)
loaded_model.eval()

print("Model and tokenizer loaded for inference.")

# Assume you have a new DataFrame with 'policy_id', 'policy_title', 'policy_summary'
# Replace 'new_policies.csv' with your actual new data file path
new_policies = pd.read_csv('new_policies.csv')

# Ensure 'policy_id', 'policy_title', 'policy_summary' exist
required_cols = ['policy_id', 'policy_title', 'policy_summary']
for col in required_cols:
    if col not in new_policies.columns:
        raise ValueError(f"Column {col} not found in new policies dataset.")

# Ensure 'policy_title' and 'policy_summary' are strings
new_policies['policy_title'] = new_policies['policy_title'].astype(str)
new_policies['policy_summary'] = new_policies['policy_summary'].astype(str)

# Combine 'policy_title' and 'policy_summary' into one text column
new_policies['text'] = new_policies['policy_title'] + ' ' + new_policies['policy_summary']

# Define function for making predictions
def predict_applicability(model, tokenizer, texts, device, threshold=0.5, batch_size=8):
    model.eval()
    predictions = []
    probabilities = []
    
    dataset = PolicyDataset(texts, labels=None, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs)
            
            preds = (probs >= threshold).int()
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    return predictions, probabilities

# Get texts and IDs for prediction
texts = new_policies['text'].tolist()
policy_ids = new_policies['policy_id'].tolist()
policy_titles = new_policies['policy_title'].tolist()
policy_summaries = new_policies['policy_summary'].tolist()

# Make predictions
predictions, probabilities = predict_applicability(
    model=loaded_model,
    tokenizer=loaded_tokenizer,
    texts=texts,
    device=device
)

# Convert predictions and probabilities to DataFrame
predictions_df = pd.DataFrame(predictions, columns=label_cols)
probabilities_df = pd.DataFrame(probabilities, columns=[f'{col}_prob' for col in label_cols])

# Map predictions to Yes/No
for col in label_cols:
    predictions_df[col] = predictions_df[col].map({1: 'Yes', 0: 'No'})

# Combine results into a single DataFrame
results = pd.DataFrame({
    'policy_id': policy_ids,
    'policy_title': policy_titles,
    'policy_summary': policy_summaries
})

results = pd.concat([results, predictions_df, probabilities_df], axis=1)

# Rearrange columns for clarity
cols = ['policy_id', 'policy_title', 'policy_summary']
for col in label_cols:
    cols.append(col)
    cols.append(f'{col}_prob')
results = results[cols]

# Display the results
print(results.head())

# Save results to a TXT file with the desired format
with open('policy_predictions.txt', 'w') as f:
    for idx, row in results.iterrows():
        f.write(f"Policy ID: {row['policy_id']}\n")
        f.write(f"Title: {row['policy_title']}\n")
        f.write(f"Summary: {row['policy_summary']}\n")
        for col in label_cols:
            prediction = row[col]
            probability = row[f'{col}_prob']
            f.write(f"{col}: {prediction} ({probability:.2f})\n")
        f.write("\n")

print("Predictions saved to 'policy_predictions.txt'.")
