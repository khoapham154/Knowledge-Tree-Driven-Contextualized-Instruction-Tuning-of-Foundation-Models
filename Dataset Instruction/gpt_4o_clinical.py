import os
import json
import requests
import nibabel as nib
import numpy as np
import torch
from PIL import Image

# Set API Key (replace with your valid key)
OPENAI_API_KEY = ""

# API Endpoints
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_UPLOAD_URL = "https://api.openai.com/v1/files"

# MRI Image Paths (update paths as needed)
IMAGE_FOLDERS = {
    "FLAIR": "your_path/FLAIR",
    "T1": "your_path/T1",
    "T2": "your_path/T2"
}

# Temporary output folder for PNG images
OUTPUT_IMAGE_FOLDER = "your_path"
os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)


def load_and_preprocess_nii_image(file_path):
    """Loads a .nii.gz file, extracts the middle slice, rotates it, and saves it as PNG."""
    if not os.path.exists(file_path):
        return None

    nii_image = nib.load(file_path)
    nii_canonical = nib.as_closest_canonical(nii_image)
    image_data = nii_canonical.get_fdata()

    # Extract middle axial slice for 3D image; otherwise use directly.
    if len(image_data.shape) == 3:
        mid = image_data.shape[2] // 2
        slice_data = image_data[:, :, mid]
    else:
        slice_data = image_data

    min_val, max_val = np.min(slice_data), np.max(slice_data)
    normalized = ((slice_data - min_val) / (max_val - min_val) * 255).astype(np.uint8) if max_val > min_val else np.zeros_like(slice_data)

    pil_image = Image.fromarray(normalized)
    rotated_image = pil_image.rotate(90, expand=True)
    output_image_path = os.path.join(OUTPUT_IMAGE_FOLDER, os.path.basename(file_path).replace(".nii.gz", ".png"))
    rotated_image.save(output_image_path)
    return output_image_path

def upload_image_to_openai(image_path):
    """Uploads a PNG image to OpenAI and returns a file ID for future use."""
    if not os.path.exists(image_path):
        return None

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    with open(image_path, "rb") as image_file:
        files = {"file": image_file, "purpose": (None, "vision")}
        response = requests.post(OPENAI_UPLOAD_URL, headers=headers, files=files)

    if response.status_code == 200:
        return response.json().get("id")
    else:
        return None

def find_images_for_source(sourceid):
    images = {}
    for modality, folder in IMAGE_FOLDERS.items():
        filename = f"{sourceid}_{modality}.nii.gz"
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            images[modality] = filepath
    return images

def generate_diagnostic_question(correct_answer, fig_caption, annotation_context):
    """
    Generates a diagnostic question and a gold standard answer using the OpenAI API.
    The prompt now only includes the figure caption and annotation tree context.
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt_text = (
        "You are a medical AI assistant tasked with generating a diagnostic question and a gold standard answer.\n\n"
        "Inputs:\n"
        # f" - Figure Caption: {fig_caption}\n"
        f" - Benchmark Diagnosis (for reference): {correct_answer}\n\n"
        # f" - Annotation Tree Context: {annotation_tree_text}\n\n"
        "Instructions:\n"
        # "1. Generate a diagnostic question that instructs a diagnostic model to analyze the MRI image and the provided figure caption.\n"
        "1. Generate a diagnostic question that instructs a diagnostic model to analyze the MRI image and the provided MRI report.\n"
        "2. Then, generate a gold standard answer to that question based on all the available information.\n\n"
        "Please clearly label your output as follows:\n"
        "Diagnostic Question: <your diagnostic question here>\n\n"
        "Gold Standard Answer: <your detailed answer here>\n\n"
        "Use clear, clinical language."
    )

    data = {
        "model": "gpt-4o",  # Adjust model if needed.
        "messages": [
            {"role": "system", "content": "You are a medical AI generating diagnostic questions with gold standard answers."},
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": 200
    }

    response = requests.post(OPENAI_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        generated_output = response.json()["choices"][0]["message"]["content"]
        return generated_output
    else:
        return prompt_text  # Fallback if API fails

def create_qa_json(input_json_path, output_json_path):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qa_list = []
    question_counter = 0

    for entry in data:
        # Use subject_id if available; otherwise fallback to sourceid.
        sourceid = entry.get("subject_id", entry.get("sourceid", "unknown"))
        modality_images = find_images_for_source(sourceid)
        fig_caption = entry.get("text", "")
        
        for modality, nii_path in modality_images.items():
            png_path = load_and_preprocess_nii_image(nii_path)
            if png_path:
                image_id = upload_image_to_openai(png_path)
                if image_id:
                    for obj in entry.get("denotations", []):
                        correct_answer = obj.get("obj", "")
                        annotation_context = None
                        generated_output = generate_diagnostic_question(
                            correct_answer,
                            fig_caption,
                            annotation_context
                        )
                        # Expect output to have two labeled sections: "Diagnostic Question:" and "Gold Standard Answer:"
                        split_text = generated_output.split("Gold Standard Answer:")
                        question_part = split_text[0].strip()
                        gold_answer = split_text[1].strip() if len(split_text) > 1 else ""
                        
                        qa_entry = {
                            "question_id": question_counter,
                            "image": png_path,
                            "pair_id": sourceid,
                            "text": question_part,
                            "gold_answer": gold_answer,
                            "correct_answer": correct_answer,
                            "fig_caption": fig_caption
                        }
                        qa_list.append(qa_entry)
                        question_counter += 1
                        
                        # Print the first three QA entries immediately for inspection.
                        if question_counter <= 3:
                            print("QA Entry:")
                            print(json.dumps(qa_entry, indent=2))
                            print("-" * 40)
                        print(f"Question generated successfully for sample {sourceid} (modality: {modality}).")

    with open(output_json_path, "w", encoding="utf-8") as out_f:
        json.dump(qa_list, out_f, indent=4)
    print(f"\nQA file successfully created at: {output_json_path}")

if __name__ == "__main__":
    input_json_path = "Your path"
    output_json_path = "Your path"
    create_qa_json(input_json_path, output_json_path)
