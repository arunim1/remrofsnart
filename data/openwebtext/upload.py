from huggingface_hub import HfApi

api = HfApi()  # make sure you `huggingface-cli login` first
api.upload_file(
    path_or_fileobj="./train.bin",  # local path
    path_in_repo="train.bin",  # where it will live in the repo
    repo_id="arunim1/openwebtext",
    repo_type="dataset",  # or "model"
)

api.upload_file(
    path_or_fileobj="./val.bin",  # local path
    path_in_repo="val.bin",  # where it will live in the repo
    repo_id="arunim1/openwebtext",
    repo_type="dataset",  # or "model"
)
