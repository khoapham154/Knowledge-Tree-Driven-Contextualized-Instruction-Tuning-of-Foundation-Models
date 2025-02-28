import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import open_clip

# -----------------------------------------
# Configuration (single source of truth)
# -----------------------------------------
config = {
    "learning_rate": 1e-4,
    "batch_size": 8,
    "epochs": 50,
    "num_folds": 5,
    "max_text_length": 256,         # maximum token length; will be passed as context_length
    "vocab_size": 49408,            # default vocabulary size for the CLIP tokenizer (adjust if needed)
    "embed_dim": 512,               # embedding dimension for text tokens
    "experiment_name": "text_only_experiment_1"
}

# -----------------------------------------
# Device Setup
# -----------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------------------
# Load JSON Metadata Helper Function
# -----------------------------------------
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# -----------------------------------------
# Load the Open-CLIP Tokenizer (text only)
# -----------------------------------------
tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

# -----------------------------------------
# Text Tokenization Helper Function
# -----------------------------------------
def tokenize_report(report, tokenizer, max_length=256):
    """
    Tokenizes a report using the provided tokenizer.
    The tokenization is done with a fixed context_length (max_length).
    If the report is empty, an empty string is used.
    """
    if not report:
        report = ""
    # The tokenizer expects a list of strings and returns a list of token sequences.
    tokens = tokenizer([report], context_length=max_length)
    # Ensure the tokens are returned as a torch.LongTensor.
    if not torch.is_tensor(tokens[0]):
        tokens = torch.tensor(tokens[0], dtype=torch.long)
    else:
        tokens = tokens[0].long()
    return tokens

# -----------------------------------------
# Text-Only Dataset
# -----------------------------------------
class TextOnlyDataset(Dataset):
    def __init__(self, metadata, tokenizer, max_length=256):
        """
        metadata: list of JSON entries.
        tokenizer: the Open-CLIP tokenizer.
        max_length: maximum token length (will be passed as context_length).
        """
        self.metadata = metadata
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        entry = self.metadata[idx]
        report = entry.get("report", "")
        tokens = tokenize_report(report, self.tokenizer, self.max_length)
        label = entry["outcome"]  # Outcome label (e.g., 0, 1, or more classes)
        return tokens, label

# -----------------------------------------
# Collate Function for Text-Only Dataset
# -----------------------------------------
def custom_collate_fn_text_only(batch):
    """
    Pads token sequences in the batch to the maximum length in that batch.
    """
    tokens_list, labels = zip(*batch)
    tokens_list = list(tokens_list)
    max_len = max(t.size(0) for t in tokens_list)
    padded_tokens = torch.zeros((len(tokens_list), max_len), dtype=torch.long)
    for i, t in enumerate(tokens_list):
        padded_tokens[i, :t.size(0)] = t
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_tokens, labels

# -----------------------------------------
# Text-Only Classifier Model
# -----------------------------------------
class TextOnlyClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Simple architecture: average token embeddings and pass through an MLP.
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, reports):
        # reports: [B, L] where L is the sequence length.
        x = self.embedding(reports)        # [B, L, embed_dim]
        x = x.mean(dim=1)                  # [B, embed_dim] (average over sequence length)
        logits = self.fc(x)                # [B, num_classes]
        return logits

# -----------------------------------------
# Training Loop for Text-Only Classification
# -----------------------------------------
def train_text_only(model, dataloader, criterion, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for reports, labels in dataloader:
            reports = reports.to(device)
            labels = labels.to(device).long()
            optimizer.zero_grad()
            logits = model(reports)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# -----------------------------------------
# Evaluation Function for Text-Only Classification
# -----------------------------------------
def evaluate_text_only(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for reports, labels in dataloader:
            reports = reports.to(device)
            labels = labels.to(device).long()
            logits = model(reports)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    try:
        all_labels_one_hot = F.one_hot(torch.tensor(all_labels), num_classes=logits.shape[1]).numpy()
        auc = roc_auc_score(all_labels_one_hot, all_probs, multi_class='ovr')
    except Exception as e:
        print("Error computing AUC:", e)
        auc = None
    print(f"Test Metrics -- Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc if auc is not None else 'N/A'}")
    return accuracy, precision, recall, auc

# -----------------------------------------
# Preprocess Fold Data Function for Text-Only Experiment
# -----------------------------------------
def preprocess_fold_data_text_only(fold_num):
    """
    Loads train and test JSON metadata for the given fold.
    Assumes that each JSON entry has a "report" field and an "outcome".
    """
    train_json_path = os.path.join(fold_json_dir, f"fold_{fold_num}_train.json")
    test_json_path = os.path.join(fold_json_dir, f"fold_{fold_num}_test.json")
    train_metadata = load_json(train_json_path)
    test_metadata = load_json(test_json_path)
    return train_metadata, test_metadata

# -----------------------------------------
# Paths and Experiment Folder Setup
# -----------------------------------------
fold_json_dir = "your_path"
output_dir = "your_path"
experiment_name = config["experiment_name"]  # e.g., "text_only_experiment_1"
results_dir = os.path.join(output_dir, experiment_name)
os.makedirs(results_dir, exist_ok=True)

# -----------------------------------------
# 5-Fold Cross-Validation for Text-Only Classification
# -----------------------------------------
num_folds = config["num_folds"]
fold_results = []
for fold_num in range(1, num_folds + 1):
    print(f"\n=== Processing Fold {fold_num} ===")
    train_metadata, test_metadata = preprocess_fold_data_text_only(fold_num)
    
    # Create text-only datasets.
    train_dataset = TextOnlyDataset(train_metadata, tokenizer, config["max_text_length"])
    test_dataset = TextOnlyDataset(test_metadata, tokenizer, config["max_text_length"])
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              collate_fn=custom_collate_fn_text_only)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False,
                             collate_fn=custom_collate_fn_text_only)
    
    # Initialize the text-only classifier.
    text_model = TextOnlyClassifier(config["vocab_size"], config["embed_dim"], num_classes=4).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(text_model.parameters(), lr=config["learning_rate"])
    
    print(f"Training Fold {fold_num} (Text-Only)...")
    train_text_only(text_model, train_loader, criterion, optimizer, config["epochs"], device)
    
    print(f"Evaluating Fold {fold_num} (Text-Only)...")
    acc, prec, rec, auc = evaluate_text_only(text_model, test_loader, device)
    fold_results.append({
        "fold": fold_num,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "auc": auc
    })
    
    # Save the model for this fold.
    torch.save(text_model.state_dict(), os.path.join(results_dir, f"text_only_model_fold_{fold_num}.pt"))

print("\n=== 5-Fold Cross-Validation Results (Text-Only) ===")
for result in fold_results:
    print(f"Fold {result['fold']}: Accuracy = {result['accuracy']:.4f}, Precision = {result['precision']:.4f}, "
          f"Recall = {result['recall']:.4f}, AUC = {result['auc'] if result['auc'] is not None else 'N/A'}")

results_path = os.path.join(results_dir, "fold_results.json")
with open(results_path, "w") as f:
    json.dump(fold_results, f, indent=4)
print(f"Results saved to: {results_path}")

# Additionally, store the configuration automatically.
config_path = os.path.join(results_dir, "config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=4)
print(f"Configuration saved to: {config_path}")
