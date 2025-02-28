import os
import json
import torch
import torch.nn.functional as F  # for one_hot and other functions
import nibabel as nib
import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import open_clip

# -----------------------------------------
# Configuration
# -----------------------------------------
config = {
    "learning_rate": 1e-4,
    "batch_size": 64,
    "epochs": 10,
    "lambda_thresh": 1.0,
    "lambda_triplet": 1.0,
    "steepness_factor": 5.0,
    "num_folds": 5,
    "experiment_name": "contrastive_loss_triplet_training_only"
}

# -----------------------------------------
# Device Setup
# -----------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------------------
# Load Model and Preprocessing (using BiomedCLIP)
# -----------------------------------------
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
model = model.to(device)

# -----------------------------------------
# Contrastive Loss (Defined but not used; using compute_triplet_loss instead)
# -----------------------------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# -----------------------------------------
# Utility Functions
# -----------------------------------------
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def check_image_dimensions(metadata):
    two_d_count = 0
    three_d_count = 0
    for entry in metadata:
        for modality, path in entry['images'].items():
            if os.path.exists(path):
                image = nib.load(path)
                if len(image.shape) == 2:
                    two_d_count += 1
                elif len(image.shape) == 3:
                    three_d_count += 1
    print(f"2D Images: {two_d_count}, 3D Images: {three_d_count}")
    return two_d_count, three_d_count

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
#     if not file_path or not os.path.exists(file_path):
#         print("Missing file path, using placeholder image")
#         return [torch.rand((3, 224, 224))]
    
#     nii_image = nib.load(file_path)
#     image_data = nii_image.get_fdata()
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

def tokenize_report(report, tokenizer, max_length=256):
    if not report:
        report = ""
    tokens = tokenizer([report], context_length=max_length)
    return tokens[0]

def preprocess_clinical_data(metadata, train_columns=None, tokenizer=None, max_length=256):
    clinical_features = []
    outcomes = []
    reports = []
    drug_categories = set()
    for entry in metadata:
        asm = entry["clinical_data"].get("asm", None)
        if asm:
            drug_categories.add(asm.lower())
    drug_categories = sorted(drug_categories)

    for entry in metadata:
        clinical_data = entry.get("clinical_data", {})
        asm = clinical_data.get("asm", None)
        asm_encoding = {drug: 0 for drug in drug_categories}
        if asm and asm.lower() in asm_encoding:
            asm_encoding[asm.lower()] = 1
        clinical_features.append(list(asm_encoding.values()))
        outcomes.append(entry["outcome"])
        reports.append(entry.get("report", ""))
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(clinical_features)
    features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    outcomes_tensor = torch.tensor(outcomes, dtype=torch.long)
    return features_tensor, outcomes_tensor, reports, drug_categories

# -----------------------------------------
# Dataset Class (No Instruction Tuning)
# -----------------------------------------
class ASMDataset(Dataset):
    def __init__(self, metadata, preprocess, clinical_data, outcomes, reports):
        self.metadata = metadata
        self.preprocess = preprocess
        self.clinical_data = clinical_data
        self.outcomes = outcomes
        # Use original report text tokenization
        self.reports = [tokenize_report(report, tokenizer, max_length=256) for report in reports]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        entry = self.metadata[idx]
        clinical = self.clinical_data[idx]
        label = self.outcomes[idx]
        report = self.reports[idx]
        def safe_load_image(image_path):
            if image_path and os.path.exists(image_path):
                return load_and_preprocess_image_v2(image_path, self.preprocess)
            else:
                return [torch.rand((3, 224, 224))]
        flair_img = safe_load_image(entry['images'].get('FLAIR'))
        t1_img = safe_load_image(entry['images'].get('T1'))
        t2_img = safe_load_image(entry['images'].get('T2'))
        return (flair_img, t1_img, t2_img, clinical, report), label

# -----------------------------------------
# Collate Function for Dynamic Padding
# -----------------------------------------
def custom_collate_fn(batch):
    flair_imgs, t1_imgs, t2_imgs, clinical_data, reports, labels = [], [], [], [], [], []
    max_flair_slices = 0
    max_t1_slices = 0
    max_t2_slices = 0

    for (flair_img, t1_img, t2_img, clinical, report), label in batch:
        max_flair_slices = max(max_flair_slices, len(flair_img))
        max_t1_slices = max(max_t1_slices, len(t1_img))
        max_t2_slices = max(max_t2_slices, len(t2_img))
        flair_imgs.append(torch.stack(flair_img))
        t1_imgs.append(torch.stack(t1_img))
        t2_imgs.append(torch.stack(t2_img))
        clinical_data.append(clinical)
        reports.append(report)
        labels.append(label)

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

    max_report_length = max(r.size(0) for r in reports)
    padded_reports = torch.zeros((len(reports), max_report_length), dtype=torch.float32)
    for i, report in enumerate(reports):
        padded_reports[i, :report.size(0)] = report

    clinical_data = torch.stack(clinical_data)
    labels = torch.tensor(labels)
    return (flair_imgs, t1_imgs, t2_imgs, clinical_data, padded_reports), labels

# -----------------------------------------
# Model Definition (Binary Classification, with Triplet Loss)
# -----------------------------------------
# Here we set num_classes = 2 for binary outcomes.
class VisionTransformerBaseline(nn.Module):
    def __init__(self, vit_model, clinical_input_dim, text_input_dim, num_classes=2):
        super().__init__()
        self.flair_encoder = vit_model.visual   # Encoder for FLAIR
        self.t1_encoder = vit_model.visual        # Encoder for T1
        self.t2_encoder = vit_model.visual        # Encoder for T2

        def extract_features(encoder, images):
            if isinstance(images, list):
                images = torch.stack(images)
            collapsed_images = images.mean(dim=1)
            features = encoder(collapsed_images)
            return features
        self.extract_features = extract_features

        self.clinical_fc = nn.Sequential(
            nn.Linear(clinical_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.text_fc = nn.Sequential(
            nn.Linear(text_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Combine image features (512 each from three modalities) and clinical/text features (128 each)
        combined_feature_dim = 512 * 3 + 128 * 2
        self.hidden_fc = nn.Sequential(
            nn.Linear(combined_feature_dim, 256),
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

    def forward(self, flair_imgs, t1_imgs, t2_imgs, clinical_data, report):
        flair_features = self.extract_features(self.flair_encoder, flair_imgs)
        t1_features = self.extract_features(self.t1_encoder, t1_imgs)
        t2_features = self.extract_features(self.t2_encoder, t2_imgs)
        clinical_features = self.clinical_fc(clinical_data)
        text_features = self.text_fc(report.float())
        combined_features = torch.cat([flair_features, t1_features, t2_features, clinical_features, text_features], dim=1)
        hidden_features = self.hidden_fc(combined_features)
        logits = self.classifier(hidden_features)
        probabilities = self.sigmoid(logits)
        predictions = torch.sigmoid(self.steepness_factor * (probabilities - self.threshold))
        return logits, predictions, hidden_features

# -----------------------------------------
# Triplet Loss Helper Function (for metric learning)
# -----------------------------------------
def compute_triplet_loss(embeddings, labels, margin=1.0):
    anchors = []
    positives = []
    negatives = []
    for i in range(len(labels)):
        anchor = embeddings[i]
        label = labels[i]
        pos_indices = (labels == label).nonzero(as_tuple=False).flatten()
        neg_indices = (labels != label).nonzero(as_tuple=False).flatten()
        pos_indices = pos_indices[pos_indices != i]
        if len(pos_indices) > 0 and len(neg_indices) > 0:
            pos = embeddings[pos_indices[0]]
            neg = embeddings[neg_indices[0]]
            anchors.append(anchor)
            positives.append(pos)
            negatives.append(neg)
    if len(anchors) == 0:
        return torch.tensor(0.0, device=embeddings.device)
    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
    loss = triplet_loss_fn(anchors, positives, negatives)
    return loss

# -----------------------------------------
# Training Loop (Classification + Triplet Loss + BCE on predictions)
# -----------------------------------------
def train_model(model, dataloader, criterion, optimizer, epochs=config["epochs"], device=device, early_stopping_patience=5):
    lambda_thresh = config["lambda_thresh"]
    lambda_triplet = config["lambda_triplet"]
    best_loss = float('inf')
    patience_counter = 0
    bce_loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, ((flair_imgs, t1_imgs, t2_imgs, clinical_data, report), labels) in enumerate(dataloader):
            flair_imgs = [img.to(device) for img in flair_imgs]
            t1_imgs = [img.to(device) for img in t1_imgs]
            t2_imgs = [img.to(device) for img in t2_imgs]
            clinical_data = clinical_data.to(device)
            report = report.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            logits, predictions, embeddings = model(flair_imgs, t1_imgs, t2_imgs, clinical_data, report)
            loss_class = criterion(logits, labels)
            one_hot_labels = F.one_hot(labels, num_classes=logits.shape[1]).float()
            loss_thresh = bce_loss_fn(predictions, one_hot_labels)
            loss_triplet = compute_triplet_loss(embeddings, labels, margin=1.0)
            loss = loss_class + lambda_thresh * loss_thresh + lambda_triplet * loss_triplet 
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
                print("Early stopping triggered. Terminating training.")
                break

# -----------------------------------------
# Evaluation Function with ROC AUC (Binary)
# -----------------------------------------
def evaluate_model(model, dataloader, device=device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch_idx, ((flair_imgs, t1_imgs, t2_imgs, clinical_data, report), labels) in enumerate(dataloader):
            flair_imgs = [img.to(device) for img in flair_imgs]
            t1_imgs = [img.to(device) for img in t1_imgs]
            t2_imgs = [img.to(device) for img in t2_imgs]
            clinical_data = clinical_data.to(device)
            report = report.to(device)
            labels = labels.to(device).long()

            logits, _, _ = model(flair_imgs, t1_imgs, t2_imgs, clinical_data, report)
            probabilities = torch.softmax(logits, dim=1)
            preds = torch.argmax(probabilities, dim=1)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probabilities.cpu().numpy())
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    # Extract probability for class 1 (seizure-free)
    y_prob = all_probs[:, 1]
    auc = roc_auc_score(all_labels, y_prob)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    print(f"\nTest Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, ROC AUC: {auc:.4f}")
    return accuracy, precision, recall, auc

# -----------------------------------------
# Preprocess Fold Data Function
# -----------------------------------------
def preprocess_fold_data(fold_num, fold_json_dir):
    train_json_path = os.path.join(fold_json_dir, f"fold_{fold_num}_train.json")
    test_json_path = os.path.join(fold_json_dir, f"fold_{fold_num}_test.json")
    train_metadata = load_json(train_json_path)
    test_metadata = load_json(test_json_path)
    train_clinical, train_outcomes, train_reports, train_columns = preprocess_clinical_data(
        train_metadata, tokenizer=tokenizer
    )
    test_clinical, test_outcomes, test_reports, _ = preprocess_clinical_data(
        test_metadata, train_columns=train_columns, tokenizer=tokenizer
    )
    return (train_metadata, test_metadata, train_clinical, train_outcomes, train_reports,
            test_clinical, test_outcomes, test_reports)

# -----------------------------------------
# Paths and Experiment Setup
# -----------------------------------------
fold_json_dir = "your_path" 
output_dir = "your_path"
experiment_name = config["experiment_name"]
results_dir = os.path.join(output_dir, experiment_name)
os.makedirs(results_dir, exist_ok=True)

# -----------------------------------------
# 5-Fold Cross-Validation Training
# -----------------------------------------
fold_results = []
num_folds = config["num_folds"]
for fold_num in range(1, num_folds + 1):
    print(f"\n=== Processing Fold {fold_num} ===")
    (train_metadata, test_metadata,
     train_clinical, train_outcomes, train_reports,
     test_clinical, test_outcomes, test_reports) = preprocess_fold_data(fold_num, fold_json_dir)
    
    train_dataset = ASMDataset(train_metadata, preprocess_train, train_clinical, train_outcomes, train_reports)
    test_dataset = ASMDataset(test_metadata, preprocess_val, test_clinical, test_outcomes, test_reports)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=custom_collate_fn)

    # Initialize model with num_classes=2 for binary classification.
    baseline_model = VisionTransformerBaseline(
        model, 
        clinical_input_dim=train_clinical.shape[1], 
        text_input_dim=train_dataset.reports[0].shape[0], 
        num_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=config["learning_rate"])

    print(f"Training Fold {fold_num}...")
    train_model(baseline_model, train_loader, criterion, optimizer, epochs=config["epochs"], device=device)

    print(f"Evaluating Fold {fold_num}...")
    acc, prec, rec, auc = evaluate_model(baseline_model, test_loader, device=device)
    fold_results.append({
        "fold": fold_num,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "auc": auc
    })
    torch.save(baseline_model.state_dict(), os.path.join(results_dir, f"baseline_model_fold_{fold_num}.pt"))

print("\n=== 5-Fold Cross-Validation Results ===")
for result in fold_results:
    print(f"Fold {result['fold']}: Accuracy = {result['accuracy']:.4f}, Precision = {result['precision']:.4f}, Recall = {result['recall']:.4f}, ROC AUC = {result['auc']:.4f}")

results_path = os.path.join(results_dir, "fold_results.json")
with open(results_path, "w") as f:
    json.dump(fold_results, f, indent=4)
print(f"Results saved to: {results_path}")
