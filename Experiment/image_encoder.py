import os
import json
import torch
import torch.nn.functional as F  # for one_hot and other functions
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import open_clip

# -----------------------------------------
# Configuration (single source of truth)
# -----------------------------------------
config = {
    "learning_rate": 1e-4,
    "batch_size": 16,
    "epochs": 50,
    "num_folds": 5,
    "experiment_name": "image_encoder_only_experiment_1"  # Change this for different experiments.
}

# -----------------------------------------
# Device Setup
# -----------------------------------------
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------------------
# Load the Open-CLIP model and transforms
# -----------------------------------------
# Here we use the same pretrained vision encoder as before.
model_clip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
# (The tokenizer is not used in the image-only experiment.)
model_clip = model_clip.to(device)

# -----------------------------------------
# Helper Function: Load JSON Metadata
# -----------------------------------------
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# -----------------------------------------
# Image Preprocessing with Orientation Standardization
# -----------------------------------------
def load_and_preprocess_image_v2(file_path, preprocess, num_slices=3):
    if not file_path or not os.path.exists(file_path):
        print("Missing file path, using placeholder image")
        return [torch.rand((3, 224, 224))]  # Placeholder tensor

    try:
        # Load NIfTI image
        nii_image = nib.load(file_path)

        # Convert to RAS+ orientation
        nii_image_ras = nib.as_closest_canonical(nii_image)
        image_data = nii_image_ras.get_fdata()

        # Normalize image data to range 0-255
        image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 255
        image_data = image_data.astype(np.uint8)

        # Rotate 90 degrees counterclockwise
        image_data = np.rot90(image_data)

        slices = []
        if len(image_data.shape) == 3:  # 3D Image
            slice_positions = np.linspace(0, image_data.shape[2] - 1, num=num_slices, dtype=int)
            slices = [preprocess(Image.fromarray(image_data[:, :, pos])) for pos in slice_positions]
        elif len(image_data.shape) == 2:  # 2D Image
            slices = [preprocess(Image.fromarray(image_data)) for _ in range(num_slices)]

        # Ensure the number of slices matches `num_slices`
        if len(slices) < num_slices:
            slices += [torch.zeros((3, 224, 224))] * (num_slices - len(slices))
        elif len(slices) > num_slices:
            slices = slices[:num_slices]

        return slices

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return [torch.rand((3, 224, 224))]  # Fallback placeholder

# def load_and_preprocess_image_v2(file_path, preprocess, num_slices=3):
#     if not file_path:
#         print("Missing file path")
#         return [torch.rand((3, 224, 224))]  # Placeholder if missing
#     nii_image = nib.load(file_path)
#     image_data = nii_image.get_fdata()
#     # Normalize to 0-255:
#     image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 255
#     image_data = image_data.astype(np.uint8)
#     slices = []
#     if len(image_data.shape) == 3:
#         slice_positions = np.linspace(0, image_data.shape[2] - 1, num=num_slices, dtype=int)
#         slices = [preprocess(Image.fromarray(image_data[:, :, pos])) for pos in slice_positions]
#     elif len(image_data.shape) == 2:
#         slices = [preprocess(Image.fromarray(image_data)) for _ in range(num_slices)]
#     if len(slices) < num_slices:
#         slices += [torch.zeros((3, 224, 224))] * (num_slices - len(slices))
#     elif len(slices) > num_slices:
#         slices = slices[:num_slices]
#     return slices

# -----------------------------------------
# Dataset for Image-Only Classification
# -----------------------------------------
class ImageOnlyDataset(Dataset):
    def __init__(self, metadata, preprocess):
        """
        metadata: list of JSON entries.
        preprocess: image preprocessing function (e.g., preprocess_train or preprocess_val).
        """
        self.metadata = metadata
        self.preprocess = preprocess

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        entry = self.metadata[idx]
        # Helper: safely load image from a file path.
        def safe_load_image(image_path):
            if image_path and os.path.exists(image_path):
                return load_and_preprocess_image_v2(image_path, self.preprocess)
            else:
                return [torch.rand((3, 224, 224))]
        # Load each modality.
        flair_img = safe_load_image(entry['images'].get('FLAIR'))
        t1_img = safe_load_image(entry['images'].get('T1'))
        t2_img = safe_load_image(entry['images'].get('T2'))
        label = entry["outcome"]  # ASM outcome (e.g., 0 or 1)
        return (flair_img, t1_img, t2_img), label

# -----------------------------------------
# Collate Function for Image-Only Dataset
# -----------------------------------------
def custom_collate_fn_image_only(batch):
    flair_imgs, t1_imgs, t2_imgs, labels = [], [], [], []
    max_flair_slices = 0
    max_t1_slices = 0
    max_t2_slices = 0
    for (flair_img, t1_img, t2_img), label in batch:
        max_flair_slices = max(max_flair_slices, len(flair_img))
        max_t1_slices = max(max_t1_slices, len(t1_img))
        max_t2_slices = max(max_t2_slices, len(t2_img))
        flair_imgs.append(torch.stack(flair_img))
        t1_imgs.append(torch.stack(t1_img))
        t2_imgs.append(torch.stack(t2_img))
        labels.append(label)
    # Pad each modality along the slice dimension.
    flair_imgs_padded = [
        torch.cat([img, torch.zeros((max_flair_slices - img.size(0), 3, 224, 224))], dim=0)
        if img.size(0) < max_flair_slices else img for img in flair_imgs
    ]
    t1_imgs_padded = [
        torch.cat([img, torch.zeros((max_t1_slices - img.size(0), 3, 224, 224))], dim=0)
        if img.size(0) < max_t1_slices else img for img in t1_imgs
    ]
    t2_imgs_padded = [
        torch.cat([img, torch.zeros((max_t2_slices - img.size(0), 3, 224, 224))], dim=0)
        if img.size(0) < max_t2_slices else img for img in t2_imgs
    ]
    flair_imgs = torch.stack(flair_imgs_padded)
    t1_imgs = torch.stack(t1_imgs_padded)
    t2_imgs = torch.stack(t2_imgs_padded)
    labels = torch.tensor(labels)
    return (flair_imgs, t1_imgs, t2_imgs), labels

# -----------------------------------------
# Image-Only Classifier Model
# -----------------------------------------
class ImageOnlyClassifier(nn.Module):
    def __init__(self, vit_model, num_classes=4):
        """
        Uses three image encoders (FLAIR, T1, T2) from the pretrained vision model.
        """
        super().__init__()
        self.flair_encoder = vit_model.visual
        self.t1_encoder = vit_model.visual
        self.t2_encoder = vit_model.visual

        # Helper function to extract features from a batch of images:
        def extract_features(encoder, images):
            # images: tensor of shape [B, S, C, H, W] (B=batch, S=slices)
            if isinstance(images, list):
                images = torch.stack(images)
            # Average over the slice dimension.
            collapsed_images = images.mean(dim=1)
            features = encoder(collapsed_images)
            return features
        self.extract_features = extract_features

        # The output dimension of each encoder is assumed to be 512.
        combined_feature_dim = 512 * 3  # Concatenate three modalities.
        self.hidden_fc = nn.Sequential(
            nn.Linear(combined_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, flair_imgs, t1_imgs, t2_imgs):
        flair_features = self.extract_features(self.flair_encoder, flair_imgs)
        t1_features = self.extract_features(self.t1_encoder, t1_imgs)
        t2_features = self.extract_features(self.t2_encoder, t2_imgs)
        combined_features = torch.cat([flair_features, t1_features, t2_features], dim=1)
        hidden = self.hidden_fc(combined_features)
        logits = self.classifier(hidden)
        return logits

# -----------------------------------------
# Training Loop for Image-Only Classification
# -----------------------------------------
def train_image_only(model, dataloader, criterion, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (flair_imgs, t1_imgs, t2_imgs), labels in dataloader:
            flair_imgs = [img.to(device) for img in flair_imgs]
            t1_imgs = [img.to(device) for img in t1_imgs]
            t2_imgs = [img.to(device) for img in t2_imgs]
            labels = labels.to(device).long()
            optimizer.zero_grad()
            logits = model(flair_imgs, t1_imgs, t2_imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# -----------------------------------------
# Evaluation Function for Image-Only Classification
# -----------------------------------------
def evaluate_image_only(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for (flair_imgs, t1_imgs, t2_imgs), labels in dataloader:
            flair_imgs = [img.to(device) for img in flair_imgs]
            t1_imgs = [img.to(device) for img in t1_imgs]
            t2_imgs = [img.to(device) for img in t2_imgs]
            labels = labels.to(device).long()
            logits = model(flair_imgs, t1_imgs, t2_imgs)
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
# Preprocess Fold Data Function for Image-Only Experiment
# -----------------------------------------
def preprocess_fold_data_image_only(fold_num):
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
experiment_name = config["experiment_name"]
results_dir = os.path.join(output_dir, experiment_name)
os.makedirs(results_dir, exist_ok=True)

# -----------------------------------------
# 5-Fold Cross-Validation for Image-Only Classification
# -----------------------------------------
num_folds = config["num_folds"]
fold_results = []
for fold_num in range(1, num_folds + 1):
    print(f"\n=== Processing Fold {fold_num} ===")
    train_metadata, test_metadata = preprocess_fold_data_image_only(fold_num)
    
    # Create datasets using the image-only dataset class.
    train_dataset = ImageOnlyDataset(train_metadata, preprocess_train)
    test_dataset = ImageOnlyDataset(test_metadata, preprocess_val)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              collate_fn=custom_collate_fn_image_only)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False,
                             collate_fn=custom_collate_fn_image_only)
    
    # Initialize the image-only classifier.
    image_only_model = ImageOnlyClassifier(model_clip, num_classes=4).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(image_only_model.parameters(), lr=config["learning_rate"])
    
    print(f"Training Fold {fold_num} (Image-Only)...")
    train_image_only(image_only_model, train_loader, criterion, optimizer, config["epochs"], device)
    
    print(f"Evaluating Fold {fold_num} (Image-Only)...")
    acc, prec, rec, auc = evaluate_image_only(image_only_model, test_loader, device)
    fold_results.append({
        "fold": fold_num,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "auc": auc
    })
    
    # Save the model for this fold.
    torch.save(image_only_model.state_dict(), os.path.join(results_dir, f"image_only_model_fold_{fold_num}.pt"))

print("\n=== 5-Fold Cross-Validation Results (Image-Only) ===")
for result in fold_results:
    print(f"Fold {result['fold']}: Accuracy = {result['accuracy']:.4f}, Precision = {result['precision']:.4f}, "
          f"Recall = {result['recall']:.4f}, AUC = {result['auc'] if result['auc'] is not None else 'N/A'}")

results_path = os.path.join(results_dir, "fold_results.json")
with open(results_path, "w") as f:
    json.dump(fold_results, f, indent=4)
print(f"Results saved to: {results_path}")

# Additionally, store the configuration automatically.
hyperparams_path = os.path.join(results_dir, "config.json")
with open(hyperparams_path, "w") as f:
    json.dump(config, f, indent=4)
print(f"Configuration saved to: {hyperparams_path}")
