import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
os.environ["WANDB_DISABLED"] = "true"
import json
import torch
import nibabel as nib
import numpy as np
from PIL import Image

# =============================================================================
# Configuration
# =============================================================================
config = {
    "max_text_length": 512,
    "qa_json": "your_path/qa_data.json",
    "llava_med_path": "your_path",
    "model_name": "llava-med-v1.5-mistral-7b"
}

# =============================================================================
# Device Setup
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Load LLaVA-Med Model, Tokenizer, and Image Processor
# =============================================================================
from llava.model.builder import load_pretrained_model
model_path = "microsoft/llava-med-v1.5-mistral-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name="llava-med-v1.5-mistral-7b",
    device="cuda"
)
model = model.float()  # Ensure model parameters are full precision
model.eval()         # Set model to evaluation mode

# =============================================================================
# Utility Functions
# =============================================================================
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_and_preprocess_nii_image(file_path, image_processor):
    """
    Loads a NIfTI image, extracts a representative slice, normalizes it,
    rotates 90°, and resizes it to 384x384 (to match the model's expected input size).
    Returns a processed tensor.
    """
    if not file_path or not os.path.exists(file_path):
        # Return a placeholder tensor with shape [3, 384, 384]
        return torch.rand((3, 384, 384))
    
    # Load and reorient the NIfTI image.
    nii_image = nib.load(file_path)
    nii_canonical = nib.as_closest_canonical(nii_image)
    image_data = nii_canonical.get_fdata()
    
    # For a 3D image, pick the middle axial slice; for 2D, use directly.
    if len(image_data.shape) == 3:
        mid = image_data.shape[2] // 2
        slice_data = image_data[:, :, mid]
    else:
        slice_data = image_data

    # Normalize the slice to [0, 255]
    min_val, max_val = np.min(slice_data), np.max(slice_data)
    if max_val > min_val:
        normalized = (slice_data - min_val) / (max_val - min_val) * 255
    else:
        normalized = np.zeros_like(slice_data)
    normalized = normalized.astype(np.uint8)
    
    # Convert to a PIL Image, rotate, and resize.
    pil_image = Image.fromarray(normalized)
    rotated_image = pil_image.rotate(90, expand=True)
    resized_image = rotated_image.resize((384, 384), Image.BICUBIC)
    
    # Process the image using the image_processor.
    processed = image_processor(resized_image, return_tensors="pt")
    processed_image = processed["pixel_values"].squeeze(0)
    
    return processed_image

def create_prompt(item):
    """
    Constructs the prompt string from a QA JSON entry.
    """
    pair_id = item.get("pair_id", "unknown")
    fig_caption = item.get("fig_caption", "")
    question_text = item.get("text", "")
    prompt = (
        f"Patient ID: {pair_id}\n"
        f"Report: {fig_caption}\n"
        f"Question: {question_text}\n"
        "Answer:"
    )
    return prompt

def run_inference(sample_item):
    """
    Generates a response for a single sample.
    Returns a dict containing the prompt, generated response, and the reference answer.
    """
    prompt = create_prompt(sample_item)
    image_path = sample_item.get("image", "")
    processed_image = load_and_preprocess_nii_image(image_path, image_processor)
    processed_image = processed_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        
        outputs = model.generate(
            input_ids,
            images=processed_image,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=150  # Adjust as needed
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {
        "prompt": prompt,
        "response": generated_text,
        "reference": sample_item.get("correct_answer", "")
    }

def main():
    qa_data = load_json(config["qa_json"])
    if not qa_data:
        return

    results = []
    # Process each sample, continuing on errors.
    for sample_item in qa_data:
        try:
            result = run_inference(sample_item)
            results.append(result)
        except Exception as e:
            # If an error occurs, record it and proceed.
            results.append({
                "prompt": create_prompt(sample_item),
                "response": f"Error: {str(e)}",
                "reference": sample_item.get("correct_answer", "")
            })
    
    # Ensure the output directory exists.
    output_dir = "your_path"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "generated_answers_all_3.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
