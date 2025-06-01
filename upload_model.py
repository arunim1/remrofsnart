#!/usr/bin/env python3
"""
Script to upload the trained model to Hugging Face.
This uploads the .pt file from out-rev-openwebtext to a model repository called gpt-2-rev.
"""

from huggingface_hub import HfApi
import os

# Initialize the Hugging Face API
# Note: Make sure you've logged in with `huggingface-cli login` before running this script
api = HfApi()

# Path to the model file
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                         "out-rev-openwebtext", "ckpt.pt")

# Repository details
# Must include username in the format 'username/repo-name'
repo_id = "arunim1/gpt-2-rev"  # Using the same username as in the existing upload.py
repo_type = "model"

print(f"Uploading model from {model_path} to {repo_id}...")

# Create the repository if it doesn't exist
try:
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    print(f"Repository {repo_id} is ready.")
except Exception as e:
    print(f"Error creating repository: {e}")
    exit(1)

# Upload the model file
try:
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="model.pt",  # Name in the repository
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Model successfully uploaded to {repo_id}!")
except Exception as e:
    print(f"Error uploading model: {e}")
    exit(1)

# Optional: Add model card with basic information
model_card = """---
language: en
license: mit
tags:
  - pytorch
  - gpt
  - language-model
---

# GPT-2 Reversed Model

This is a GPT-2 model trained on reversed OpenWebText data.

## Model Details

This model was trained using the transformer architecture with reversed text from the OpenWebText dataset.
"""

try:
    with open("README.md", "w") as f:
        f.write(model_card)
    
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print("Model card uploaded.")
    os.remove("README.md")  # Clean up temporary file
except Exception as e:
    print(f"Error uploading model card: {e}")
