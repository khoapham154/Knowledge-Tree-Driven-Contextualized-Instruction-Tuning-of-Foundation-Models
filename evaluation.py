import json
import asyncio
import os
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import SemanticSimilarity
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings

# Load API Key securely
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("API key not found. Set the OPENAI_API_KEY environment variable.")

# Initialize OpenAI Embeddings
print("Initializing OpenAI Embeddings...")
openai_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Wrap the embeddings for Ragas
evaluator_embedding = LangchainEmbeddingsWrapper(openai_embeddings)

# Initialize the scorer
scorer = SemanticSimilarity(embeddings=evaluator_embedding)
print("SemanticSimilarity scorer initialized.")

# Load JSON file (process all samples)
json_file_path = "Your_JSON_File_Path.json"
print(f"Loading JSON file: {json_file_path}")
with open(json_file_path, "r") as f:
    data = json.load(f)

# Define an async function to compute scores
async def compute_scores():
    scores_list = []
    print(f"Processing {len(data)} samples...")

    for i, item in enumerate(data):
        print(f"Processing sample {i+1}/{len(data)}...")
        sample = SingleTurnSample(response=item["response"], reference=item["reference"])
        
        try:
            score = await scorer.single_turn_ascore(sample)
            scores_list.append(score)
            print(f"score: {score}")
        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")

    # Calculate average score
    avg_score = sum(scores_list) / len(scores_list) if scores_list else 0
    print(f"Average score: {avg_score}")

    # Save results to JSON file
    output_data = {
        "scores": scores_list,
        "average_score": avg_score
    }
    output_file_path = "semantic_score_notree.json"
    with open(output_file_path, "w") as outfile:
        json.dump(output_data, outfile, indent=4)
    print(f"Results saved to {output_file_path}")

# Run the async function
print("Starting async computation...")
asyncio.run(compute_scores())
print("Processing complete.")
