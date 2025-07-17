#!/usr/bin/env python
"""
Inference Pipeline for Language Generation and Classification

This script performs inference using LLaVA-Med for language generation and CombinedClassifier 
for classification. It takes MRI images (FLAIR, T1, T2), an anti-seizure medication (ASM), 
and a subject ID as input, generating a language response and a binary classification outcome.

Inputs:
- MRI images (FLAIR, T1, T2) as NIfTI or PNG files
- ASM drug (e.g., "LEV" for Levetiracetam)
- Subject ID (e.g., "ALB0005")
- Optional QA item for language generation prompt

Outputs:
- Generated language response (e.g., treatment recommendation)
- Classification results (logits, probabilities, predicted class)
"""

import os
import torch
import json
import numpy as np
from PIL import Image
import nibabel as nib
import torch.nn as nn
import torch.nn.functional as F
from llava.model.builder import load_pretrained_model

# Configuration
config = {
    "smiles_embedding_path": "your_path/smiles_embeddings.json",
    "results_dir": "your_path",
    "max_text_length": 256,
    "target_size": 224  # For classification, matches CombinedClassifier expectation
}

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load LLaVA-Med Model, Tokenizer, and Image Processor
model_path = "your_path/llava-med-v1.5-mistral-7b"
tokenizer, llava_model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name="llava-med-v1.5-mistral-7b",
    device="cuda"
)
llava_model = llava_model.float()
llava_model.eval()
vision_tower = llava_model.get_vision_tower()
print(f"Vision tower image size: {vision_tower.config.image_size}")  # Add this after loading the vision_tower

# Load SMILES Embeddings
def load_smiles_embeddings(path):
    with open(path, 'r') as f:
        return json.load(f)

smiles_embeddings = load_smiles_embeddings(config["smiles_embedding_path"])

def get_drug_embedding(asm):
    if asm and asm in smiles_embeddings:
        return np.array(smiles_embeddings[asm], dtype=np.float32)
    example_key = next(iter(smiles_embeddings))
    emb_dim = len(smiles_embeddings[example_key])
    return np.zeros(emb_dim, dtype=np.float32)

# Utility Functions for Language Generation (from llava_med_grained.py)
def load_and_preprocess_image_lang(file_path, image_processor):
    """Loads and preprocesses an image for LLaVA-Med using the provided image_processor."""
    if file_path.lower().endswith(".png"):
        try:
            pil_image = Image.open(file_path)
        except Exception as e:
            print(f"Error opening PNG {file_path}: {e}")
            return torch.zeros((3, 336, 336))  # Fallback size
        rotated_image = pil_image.rotate(90, expand=True)
        # Remove manual resize; let image_processor handle it
        processed = image_processor(rotated_image, return_tensors="pt")
        return processed["pixel_values"].squeeze(0)
    
    if not file_path or not os.path.exists(file_path):
        return torch.zeros((3, 336, 336))
    
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
        normalized = np.zeros_like(slice_data) if max_val <= min_val else \
                    (slice_data - min_val) / (max_val - min_val) * 255
        normalized = normalized.astype(np.uint8)
        pil_image = Image.fromarray(normalized)
        rotated_image = pil_image.rotate(90, expand=True)
        # Remove manual resize
        processed = image_processor(rotated_image, return_tensors="pt")
        return processed["pixel_values"].squeeze(0)
    except Exception as e:
        print(f"Error processing NIfTI {file_path}: {e}")
        return torch.zeros((3, 336, 336))

def create_prompt(item, asm="LEV"):
    """Creates a prompt for language generation including the ASM."""
    pair_id = item.get("pair_id", "unknown")
    question_text = item.get("text", "")
    return f"Patient ID: {pair_id}\nAnti-Seizure Medication: {asm}\nQuestion: Based on the MRI findings and the use of {asm}, why might the patient not be seizure-free? Provide reasoning based on the image findings.\nAnswer:"

def run_inference(llava_model, tokenizer, image_processor, sample_item, device, asm="LEV"):
    """Generates a language response using LLaVA-Med."""
    prompt = create_prompt(sample_item, asm)
    image_path = sample_item.get("image", "")
    processed_image = load_and_preprocess_image_lang(image_path, image_processor)
    processed_image = processed_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        outputs = llava_model.generate(
            input_ids,
            images=processed_image,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=150
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Utility Functions for Classification (from combined_pipeline.py)
def load_and_preprocess_image_class(file_path, image_processor, target_size=config["target_size"]):
    """Loads and preprocesses an image for CombinedClassifier (resizes to 224x224)."""
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
        normalized = np.zeros_like(slice_data) if max_val <= min_val else \
                    (slice_data - min_val) / (max_val - min_val) * 255
        normalized = normalized.astype(np.uint8)
        pil_image = Image.fromarray(normalized)
        rotated_image = pil_image.rotate(90, expand=True)
        resized_image = rotated_image.resize((target_size, target_size), Image.BICUBIC)
        processed = image_processor(resized_image, return_tensors="pt")
        return processed["pixel_values"].squeeze(0)
    except Exception as e:
        print(f"Error processing NIfTI {file_path}: {e}")
        return torch.zeros((3, target_size, target_size))

def tokenize_report(text, tokenizer, max_length=config["max_text_length"]):
    """Tokenizes text for classification."""
    text = text if text else ""
    tokens = tokenizer([text], max_length=max_length, truncation=True, return_tensors="pt")
    return tokens["input_ids"].squeeze(0)

# Define Classification Model Classes (from combined_pipeline.py)
class ImageEncoder(nn.Module):
    def __init__(self, vision_tower):
        super().__init__()
        self.vision_tower = vision_tower
        self.expected_size = 224
        try:
            if hasattr(self.vision_tower, 'config'):
                self.expected_size = getattr(self.vision_tower.config, 'image_size', 224)
            elif hasattr(self.vision_tower, 'vision_model') and hasattr(self.vision_tower.vision_model, 'config'):
                self.expected_size = getattr(self.vision_tower.vision_model.config, 'image_size', 224)
        except:
            pass
    
    def forward(self, x):
        if x.shape[2] != self.expected_size or x.shape[3] != self.expected_size:
            x = F.interpolate(x, size=(self.expected_size, self.expected_size), mode='bilinear', align_corners=False)
        with torch.no_grad():
            features = self.vision_tower(x)
            if features.dim() == 3:
                features = features[:, 0, :]
        return features

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, output_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, output_dim)
    
    def forward(self, token_ids):
        emb = self.embedding(token_ids)
        pooled = emb.mean(dim=1)
        return self.proj(pooled)

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
        
        sample_input = torch.randn(1, 3, self.image_encoder.expected_size, self.image_encoder.expected_size).to(device)
        with torch.no_grad():
            features = self.image_encoder(sample_input)
        embed_dim = features.size(-1)
        
        total_features_size = (3 * embed_dim) + 128 + 128  # 3 images + clinical + text
        self.hidden_fc = nn.Sequential(
            nn.Linear(total_features_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()
    
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
        return logits, probabilities

# Load CombinedClassifier
image_encoder = ImageEncoder(vision_tower).to(device)
vocab_size = getattr(tokenizer, "vocab_size", 30000)
text_encoder = TextEncoder(vocab_size=vocab_size).to(device)
clinical_encoder = ClinicalEncoder(input_dim=len(get_drug_embedding("LEV"))).to(device)
combined_model = CombinedClassifier(image_encoder, clinical_encoder, text_encoder).to(device)

# Load trained model state (e.g., from fold 5)
model_path = "your_path/combined_classifier_fold_5.pth"
if os.path.exists(model_path):
    combined_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    combined_model.eval()
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Inference Function
def infer_sample(mri_paths, asm="LEV", subject_id="unknown", question_text=None):
    """
    Performs inference with both language generation and classification.
    
    Args:
        mri_paths (dict): Paths to FLAIR, T1, T2 images (e.g., {"FLAIR": "...", "T1": "...", "T2": "..."})
        asm (str): Anti-seizure medication (default: "LEV")
        subject_id (str): Patient ID (e.g., "ALB0005")
        question_text (str, optional): Custom question (not used here)
    
    Returns:
        dict: Generated text and classification results
    """
    # Step 1: Language Generation
    sample_item = {
        "pair_id": subject_id,
        "text": "Based on the MRI findings and the use of {asm}, why might the patient not be seizure-free? Provide reasoning based on the image findings.",
        "image": mri_paths.get("FLAIR", "")  # Use FLAIR for language generation
    }
    generated_text = run_inference(llava_model, tokenizer, image_processor, sample_item, device, asm)
    print("Generated Language Response:", generated_text)

    # Step 2: Prepare Classification Inputs
    flair_img = load_and_preprocess_image_class(mri_paths.get("FLAIR", ""), image_processor).unsqueeze(0).to(device)
    t1_img = load_and_preprocess_image_class(mri_paths.get("T1", ""), image_processor).unsqueeze(0).to(device)
    t2_img = load_and_preprocess_image_class(mri_paths.get("T2", ""), image_processor).unsqueeze(0).to(device)
    clinical_vector = torch.tensor(get_drug_embedding(asm), dtype=torch.float32).unsqueeze(0).to(device)
    tokenized_prompt = tokenize_report(generated_text, tokenizer).unsqueeze(0).to(device)

    # Step 3: Classification
    with torch.no_grad():
        logits, probabilities = combined_model(flair_img, t1_img, t2_img, clinical_vector, tokenized_prompt)
        predicted_class = (probabilities[:, 1] > 0.5).int().item()  # 1 = seizure-free, 0 = not

    print("\nClassification Results:")
    print(f"Logits: {logits.cpu().numpy()}")
    print(f"Probabilities: {probabilities.cpu().numpy()}")
    print(f"Predicted Class (1 = Seizure-Free, 0 = Not): {predicted_class}")

    # Step 4: Print Drug Embedding
    drug_embedding = get_drug_embedding(asm)
    print(f"\nDrug Embedding for {asm}:")
    print(drug_embedding)

    return {
        "generated_text": generated_text,
        "logits": logits.cpu().numpy(),
        "probabilities": probabilities.cpu().numpy(),
        "predicted_class": predicted_class,
        "drug_embedding": drug_embedding
    }

# Main Execution
if __name__ == "__main__":
    # Sample inputs (replace with actual paths)
    sample_mri_paths = {
        "FLAIR": "your_path/FLAIR.nii.gz",
        "T1": "your_path/T1.nii.gz",
        "T2": "your_path/T2.nii.gz"
    }
    sample_asm = "LEV"
    sample_subject_id = "ALB0005"
    # sample_question = "Based on the MRI findings, what is the recommended treatment?"

    # Run inference
    result = infer_sample(sample_mri_paths, sample_asm, sample_subject_id)