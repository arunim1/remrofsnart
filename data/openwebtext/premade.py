from huggingface_hub import hf_hub_download
from pathlib import Path
import os

# Folder you want the binaries to end up in (here: the script’s dir)
HERE = Path(__file__).resolve().parent

train_path = hf_hub_download(
    repo_id="arunim1/openwebtext",
    filename="train.bin",
    repo_type="dataset",
    cache_dir=HERE          # 👈 put the cache *here*, not ~/.cache
)
print(train_path)            # …/<repo>/snapshots/<hash>/train.bin

val_path = hf_hub_download(
    repo_id="arunim1/openwebtext",
    filename="val.bin",
    repo_type="dataset",
    cache_dir=HERE
)
print(val_path)
