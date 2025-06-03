#!/usr/bin/env python3
"""
Script to download the trained model from Hugging Face.
This downloads the model.pt file from a model repository called gpt-2-rev.
"""

from huggingface_hub import hf_hub_download
import os
import shutil

# Initialize output directory
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          "out-rev-openwebtext")

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Repository details
# Must include username in the format 'username/repo-name'
repo_id = "arunim1/gpt-2-rev"
repo_type = "model"
filename = "model.pt"
local_filename = "ckpt.pt"  # Match the filename expected by the training code

print(f"Downloading model from {repo_id}...")

try:
    # Download the model file
    downloaded_model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        cache_dir="./.cache"  # Use a local cache directory
    )
    
    # Copy to the expected location with the expected filename
    target_path = os.path.join(output_dir, local_filename)
    shutil.copy(downloaded_model_path, target_path)
    
    print(f"Model successfully downloaded to {target_path}!")
    
    # Optional: Download the model card (README.md)
    try:
        readme_path = hf_hub_download(
            repo_id=repo_id,
            filename="README.md",
            repo_type=repo_type,
            cache_dir="./.cache"
        )
        print(f"Model card downloaded to {readme_path}")
    except Exception as e:
        print(f"Note: Could not download README.md: {e}")
        
except Exception as e:
    print(f"Error downloading model: {e}")
    exit(1)

print("Download complete!")
