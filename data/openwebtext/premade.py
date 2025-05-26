from huggingface_hub import hf_hub_download

# returns a path inside ~/.cache/huggingface/hub/<repo>/...
bin_path = hf_hub_download(
    repo_id="arunim1/openwebtext",
    filename="train.bin",
    repo_type="dataset",  # or "model" if you put it there
)
print(bin_path)

bin_path = hf_hub_download(
    repo_id="arunim1/openwebtext",
    filename="val.bin",
    repo_type="dataset",  # or "model" if you put it there
)
print(bin_path)
