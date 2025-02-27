#!/usr/bin/env python
"""
MRI Classification Pipeline using LLaVA Components

Inputs:
  - A NIfTI or PNG MRI image is preprocessed (normalized, rotated 90°, resized to 384x384)
    using the LLaVA image_processor.
  - The ASM drug (e.g. "LEV") is converted into a SMILES embedding (loaded from a JSON file).
  - A text prompt is obtained from a GPT-generated response (loaded from a file) using subject ID.
  
Processing:
  - Images from three modalities (FLAIR, T1, T2) are encoded via the vision tower 
    to obtain a feature vector per modality (dimension determined dynamically).
  - The clinical (SMILES) embedding is passed through a linear layer to produce a 128-dim vector.
  - The tokenized text prompt is passed through a trainable text encoder to produce a 128-dim vector.
  - All features are concatenated and passed through an MLP head for binary classification.

A 5-fold cross-validation training/evaluation loop is provided.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. Configuration and Device Setup
# ============================================================
config = {
    "learning_rate": 1e-4,
    "batch_size": 8,
    "epochs": 10,
    "num_folds": 5,
    "steepness_factor": 5.0,
    "fold_json_dir": "/mnt/khoa/baseline/dataset/json",
    "results_dir": "/mnt/khoa/baseline/results/integration",
    "gpt_answers_file": "/mnt/khoa/baseline/mri_results/GPT_generated_answers_all_1.json",
    "smiles_embedding_path": "/mnt/khoa/baseline/smiles_moler_embedding.json",
    "max_text_length": 256
}
os.makedirs(config["results_dir"], exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# 2. Load LLaVA-Med Model, Tokenizer, and Image Processor
# ============================================================
from llava.model.builder import load_pretrained_model
model_path = "microsoft/llava-med-v1.5-mistral-7b"
tokenizer, llava_model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name="llava-med-v1.5-mistral-7b",
    device="cuda"
)
llava_model = llava_model.float()
llava_model.eval()

vision_tower = llava_model.get_vision_tower()

# ============================================================
# 3. Load SMILES Embeddings and Get ASM Embedding
# ============================================================
def load_smiles_embeddings(path):
    with open(path, 'r') as f:
        return json.load(f)

smiles_embeddings = load_smiles_embeddings(config["smiles_embedding_path"])

def get_drug_embedding(asm):
    if asm and asm in smiles_embeddings:
        emb = np.array(smiles_embeddings[asm], dtype=np.float32)
        return emb
    example_key = next(iter(smiles_embeddings))
    emb_dim = len(smiles_embeddings[example_key])
    return np.zeros(emb_dim, dtype=np.float32)

# ============================================================
# 4. Load GPT-Generated Answers and Build Mapping
# ============================================================
def load_gpt_answers(gpt_file):
    with open(gpt_file, 'r', encoding='utf-8') as f:
        answers = json.load(f)
    mapping = {}
    for entry in answers:
        prompt = entry.get("prompt", "")
        response = entry.get("response", "")
        lines = prompt.splitlines()
        if lines and "Patient ID:" in lines[0]:
            subject_id = lines[0].split("Patient ID:")[1].strip()
            mapping[subject_id] = response
    return mapping

gpt_mapping = load_gpt_answers(config["gpt_answers_file"])
print(f"Loaded GPT answers for {len(gpt_mapping)} subjects.")

# ============================================================
# 5. Preprocess Clinical Data
# ============================================================
def preprocess_clinical_data(metadata):
    clinical_features = []
    outcomes = []
    subject_ids = []
    for entry in metadata:
        asm = entry.get("clinical_data", {}).get("asm", None)
        emb = get_drug_embedding(asm)
        clinical_features.append(emb)
        outcomes.append(entry["outcome"])
        subject_ids.append(entry.get("subject_id", "unknown"))
    features_tensor = torch.tensor(np.array(clinical_features), dtype=torch.float32)
    outcomes_tensor = torch.tensor(outcomes, dtype=torch.long)
    return features_tensor, outcomes_tensor, subject_ids

# ============================================================
# 6. Utility Function: Load and Preprocess an Image
# ============================================================
def load_and_preprocess_image(file_path, image_processor, target_size=224):
    if file_path.lower().endswith(".png"):
        try:
            pil_image = Image.open(file_path)
        except Exception as e:
            print(f"Error opening PNG {file_path}: {e}")
            return torch.zeros((3, target_size, target_size))
        rotated_image = pil_image.rotate(90, expand=True)
        resized_image = rotated_image.resize((target_size, target_size), Image.BICUBIC)
        processed = image_processor(resized_image, return_tensors="pt")
        return processed["pixel_values"].squeeze(0)
    
    if not file_path or not os.path.exists(file_path):
        return torch.zeros((3, target_size, target_size))
    
    try:
        nii_image = nib.load(file_path)
        nii_canonical = nib.as_closest_canonical(nii_image)
        image_data = nii_canonical.get_fdata()
        if len(image_data.shape) == 3:
            mid = image_data.shape[2] // 2
            slice_data = image_data[:, :, mid]
        else:
            slice_data = image_data
        min_val, max_val = np.min(slice_data), np.max(slice_data)
        if max_val > min_val:
            normalized = (slice_data - min_val) / (max_val - min_val) * 255
        else:
            normalized = np.zeros_like(slice_data)
        normalized = normalized.astype(np.uint8)
        pil_image = Image.fromarray(normalized)
        rotated_image = pil_image.rotate(90, expand=True)
        resized_image = rotated_image.resize((target_size, target_size), Image.BICUBIC)
        processed = image_processor(resized_image, return_tensors="pt")
        return processed["pixel_values"].squeeze(0)
    except Exception as e:
        print(f"Error processing NIfTI {file_path}: {e}")
        return torch.zeros((3, target_size, target_size))

def tokenize_report(text, tokenizer, max_length=256):
    if not text:
        text = ""
    tokens = tokenizer([text], max_length=max_length, truncation=True, return_tensors="pt")
    return tokens["input_ids"].squeeze(0)

# ============================================================
# 8. Define the ASMDataset Class
# ============================================================
class ASMDataset(Dataset):
    def __init__(self, metadata, image_processor, clinical_data, outcomes, subject_ids, gpt_mapping, tokenizer, target_size=224):
        self.metadata = metadata
        self.image_processor = image_processor
        self.clinical_data = clinical_data
        self.outcomes = outcomes
        self.subject_ids = subject_ids
        self.tokenizer = tokenizer
        self.target_size = target_size
        self.text_prompts = []
        for sid in subject_ids:
            text = gpt_mapping.get(sid, "No GPT answer available.")
            tokenized = tokenize_report(text, self.tokenizer, max_length=config["max_text_length"])
            self.text_prompts.append(tokenized)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        entry = self.metadata[idx]
        clinical = self.clinical_data[idx]
        label = self.outcomes[idx]
        text_prompt = self.text_prompts[idx]
        
        def safe_load_image(path):
            if path and os.path.exists(path):
                return load_and_preprocess_image(path, self.image_processor, target_size=self.target_size)
            else:
                return torch.zeros((3, self.target_size, self.target_size))
        
        flair_img = safe_load_image(entry["images"].get("FLAIR"))
        t1_img = safe_load_image(entry["images"].get("T1"))
        t2_img = safe_load_image(entry["images"].get("T2"))
        
        return (flair_img, t1_img, t2_img, clinical, text_prompt), label

# ============================================================
# 9. Custom Collate Function
# ============================================================
def custom_collate_fn(batch):
    flair_imgs, t1_imgs, t2_imgs, clinical_data, prompts, labels = [], [], [], [], [], []
    for (flair_img, t1_img, t2_img, clinical, prompt), label in batch:
        flair_imgs.append(flair_img)
        t1_imgs.append(t1_img)
        t2_imgs.append(t2_img)
        clinical_data.append(clinical)
        prompts.append(prompt)
        labels.append(label)
    flair_imgs = torch.stack(flair_imgs)
    t1_imgs = torch.stack(t1_imgs)
    t2_imgs = torch.stack(t2_imgs)
    max_prompt_length = max(r.size(0) for r in prompts)
    padded_prompts = torch.zeros((len(prompts), max_prompt_length), dtype=torch.long)
    for i, prompt in enumerate(prompts):
        padded_prompts[i, :prompt.size(0)] = prompt
    clinical_data = torch.stack(clinical_data)
    labels = torch.tensor(labels)
    return (flair_imgs, t1_imgs, t2_imgs, clinical_data, padded_prompts), labels

# ============================================================
# 10. Define Trainable Branches and the Combined Classifier
# ============================================================
class ImageEncoder(nn.Module):
    def __init__(self, vision_tower):
        super().__init__()
        self.vision_tower = vision_tower
        self.expected_size = 224  # Default
        try:
            if hasattr(self.vision_tower, 'config'):
                self.expected_size = getattr(self.vision_tower.config, 'image_size', 224)
            elif hasattr(self.vision_tower, 'vision_model') and hasattr(self.vision_tower.vision_model, 'config'):
                self.expected_size = getattr(self.vision_tower.vision_model.config, 'image_size', 224)
        except:
            pass
        print(f"Vision Tower expected image size: {self.expected_size}x{self.expected_size}")
        
    def forward(self, x):
        b, c, h, w = x.shape
        if h != self.expected_size or w != self.expected_size:
            x = F.interpolate(x, size=(self.expected_size, self.expected_size), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            features = self.vision_tower(x)
            # Assuming features is [B, num_patches, embed_dim], take [CLS] token (first token)
            if features.dim() == 3:
                features = features[:, 0, :]  # [B, embed_dim]
        
        return features

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, output_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, output_dim)
    def forward(self, token_ids):
        emb = self.embedding(token_ids)
        pooled = emb.mean(dim=1)
        out = self.proj(pooled)
        return out

class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.fc(x)

class CombinedClassifier(nn.Module):
    def __init__(self, image_encoder, clinical_encoder, text_encoder, num_classes=2):
        super().__init__()
        self.image_encoder = image_encoder
        self.clinical_encoder = clinical_encoder
        self.text_encoder = text_encoder
        
        # Dynamically determine image feature dimension
        sample_input = torch.randn(1, 3, self.image_encoder.expected_size, self.image_encoder.expected_size).to(device)
        with torch.no_grad():
            features = self.image_encoder(sample_input)
        embed_dim = features.size(-1)
        print(f"Determined image feature dimension: {embed_dim}")
        
        image_features_size = 3 * embed_dim
        clinical_features_size = 128
        text_features_size = 128
        total_features_size = image_features_size + clinical_features_size + text_features_size
        
        self.hidden_fc = nn.Sequential(
            nn.Linear(total_features_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.threshold = nn.Parameter(torch.tensor(0.5))
        self.steepness_factor = config["steepness_factor"]
    
    def forward(self, flair_imgs, t1_imgs, t2_imgs, clinical_data, text_tokens):
        flair_features = self.image_encoder(flair_imgs)
        t1_features = self.image_encoder(t1_imgs)
        t2_features = self.image_encoder(t2_imgs)
        clinical_features = self.clinical_encoder(clinical_data)
        text_features = self.text_encoder(text_tokens)
        combined = torch.cat([flair_features, t1_features, t2_features, clinical_features, text_features], dim=1)
        hidden = self.hidden_fc(combined)
        logits = self.classifier(hidden)
        probabilities = self.sigmoid(logits)
        predictions = torch.sigmoid(self.steepness_factor * (probabilities - self.threshold))
        return logits, predictions, hidden

# ============================================================
# 11. Build the Complete Model
# ============================================================
image_encoder = ImageEncoder(vision_tower).to(device)
vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 30000
text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=256, output_dim=128).to(device)
example_emb = get_drug_embedding("LEV")
clinical_encoder = ClinicalEncoder(input_dim=len(example_emb), output_dim=128).to(device)
combined_model = CombinedClassifier(image_encoder, clinical_encoder, text_encoder, num_classes=2).to(device)

# ============================================================
# 14. Training Loop
# ============================================================
def train_model(model, dataloader, criterion, optimizer, epochs=config["epochs"], device=device, early_stopping_patience=5):
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, ((flair_imgs, t1_imgs, t2_imgs, clinical_data, prompt), labels) in enumerate(dataloader):
            flair_imgs = flair_imgs.to(device)
            t1_imgs = t1_imgs.to(device)
            t2_imgs = t2_imgs.to(device)
            clinical_data = clinical_data.to(device)
            prompt = prompt.to(device)
            labels = labels.to(device).long()
            optimizer.zero_grad()
            logits, predictions, _ = model(flair_imgs, t1_imgs, t2_imgs, clinical_data, prompt)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

# ============================================================
# 15. Evaluation Function
# ============================================================
def evaluate_model(model, dataloader, device=device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch_idx, ((flair_imgs, t1_imgs, t2_imgs, clinical_data, prompt), labels) in enumerate(dataloader):
            flair_imgs = flair_imgs.to(device)
            t1_imgs = t1_imgs.to(device)
            t2_imgs = t2_imgs.to(device)
            clinical_data = clinical_data.to(device)
            prompt = prompt.to(device)
            labels = labels.to(device).long()
            logits, _, _ = model(flair_imgs, t1_imgs, t2_imgs, clinical_data, prompt)
            probabilities = torch.softmax(logits, dim=1)
            preds = torch.argmax(probabilities, dim=1)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probabilities.cpu().numpy())
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    auc = roc_auc_score(all_labels, all_probs[:, 1])
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    print(f"\nEvaluation: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, ROC AUC: {auc:.4f}")
    return accuracy, precision, recall, auc

# ============================================================
# 16. Preprocess Data for a Given Fold
# ============================================================
def preprocess_fold_data(fold_num, fold_json_dir):
    train_json_path = os.path.join(fold_json_dir, f"fold_{fold_num}_train.json")
    test_json_path = os.path.join(fold_json_dir, f"fold_{fold_num}_test.json")
    with open(train_json_path, 'r') as f:
        train_metadata = json.load(f)
    with open(test_json_path, 'r') as f:
        test_metadata = json.load(f)
    train_clinical, train_outcomes, train_subject_ids = preprocess_clinical_data(train_metadata)
    test_clinical, test_outcomes, test_subject_ids = preprocess_clinical_data(test_metadata)
    return train_metadata, test_metadata, train_clinical, train_outcomes, train_subject_ids, test_clinical, test_outcomes, test_subject_ids

# ============================================================
# 17. 5-Fold Cross-Validation Training and Evaluation
# ============================================================
def run_classification_cv():
    expected_size = image_encoder.expected_size
    print(f"Using image size: {expected_size}x{expected_size}")
    
    fold_results = []
    for fold_num in range(1, config["num_folds"]+1):
        print(f"\n=== Fold {fold_num} ===")
        (train_metadata, test_metadata,
         train_clinical, train_outcomes, train_subject_ids,
         test_clinical, test_outcomes, test_subject_ids) = preprocess_fold_data(fold_num, config["fold_json_dir"])
        
        train_dataset = ASMDataset(train_metadata, image_processor, train_clinical, train_outcomes, train_subject_ids, gpt_mapping, tokenizer, target_size=expected_size)
        test_dataset = ASMDataset(test_metadata, image_processor, test_clinical, test_outcomes, test_subject_ids, gpt_mapping, tokenizer, target_size=expected_size)
        
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=custom_collate_fn)
        
        model_instance = CombinedClassifier(image_encoder, clinical_encoder, text_encoder, num_classes=2).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model_instance.parameters(), lr=config["learning_rate"])
        
        print(f"Training Fold {fold_num}...")
        train_model(model_instance, train_loader, criterion, optimizer, epochs=config["epochs"], device=device)
        
        print(f"Evaluating Fold {fold_num}...")
        acc, prec, rec, auc = evaluate_model(model_instance, test_loader, device=device)
        fold_results.append({
            "fold": fold_num,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "auc": auc
        })
        model_save_path = os.path.join(config["results_dir"], f"combined_model_fold_{fold_num}.pt")
        torch.save(model_instance.state_dict(), model_save_path)
        print(f"Saved model for fold {fold_num} to {model_save_path}")
    
    print("\n=== Cross-Validation Results ===")
    for res in fold_results:
        print(f"Fold {res['fold']}: Accuracy={res['accuracy']:.4f}, Precision={res['precision']:.4f}, Recall={res['recall']:.4f}, ROC AUC={res['auc']:.4f}")
    results_path = os.path.join(config["results_dir"], "fold_results.json")
    with open(results_path, "w") as f:
        json.dump(fold_results, f, indent=4)
    print(f"Results saved to {results_path}")

# ============================================================
# 18. Optional: Single Sample Simulation
# ============================================================
def simulate_instruction_prompt(model, tokenizer, device):
    sample_subject_id = "ALB0005"
    sample_gpt = gpt_mapping.get(sample_subject_id, "No GPT answer available.")
    tokenized_prompt = tokenize_report(sample_gpt, tokenizer, max_length=config["max_text_length"])
    tokenized_prompt = tokenized_prompt.unsqueeze(0)
    dummy_image = torch.rand((3, image_encoder.expected_size, image_encoder.expected_size))
    flair_img = dummy_image.unsqueeze(0)
    t1_img = dummy_image.unsqueeze(0)
    t2_img = dummy_image.unsqueeze(0)
    clinical_vector = torch.tensor(get_drug_embedding("LEV"), dtype=torch.float32).unsqueeze(0)
    flair_img = flair_img.to(device)
    t1_img = t1_img.to(device)
    t2_img = t2_img.to(device)
    clinical_vector = clinical_vector.to(device)
    tokenized_prompt = tokenized_prompt.to(device)
    logits, predictions, _ = model(flair_img, t1_img, t2_img, clinical_vector, tokenized_prompt)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = int((probabilities[:, 1] > 0.5).item())
    print("\n=== Single Sample Simulation ===")
    print("Subject ID:", sample_subject_id)
    print("GPT Response Used as Prompt:", sample_gpt)
    print("Model Logits:", logits.detach().cpu().numpy())
    print("Model Probabilities:", probabilities.detach().cpu().numpy())
    print("Predicted Outcome (seizure-free):", predicted_class)

# ============================================================
# 19. Main Execution
# ============================================================
def main():
    print("Starting 5-Fold Cross-Validation Training and Evaluation...")
    run_classification_cv()
    
    print("\nSimulating single sample inference...")
    fold_model_path = os.path.join(config["results_dir"], f"combined_model_fold_{config['num_folds']}.pt")
    sim_model = CombinedClassifier(image_encoder, clinical_encoder, text_encoder, num_classes=2).to(device)
    sim_model.load_state_dict(torch.load(fold_model_path, map_location=device))
    simulate_instruction_prompt(sim_model, tokenizer, device)

if __name__ == '__main__':
    main()