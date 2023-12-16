"""
Download an entire specified Huggingface hub repository to a local directory.
Useful when we want to finetune a HF model, and need to load a pretrained model on all ranks.

Trying to directly use the repository URL on multiple nodes
will likely lead to being rate-throttled by Huggingface
as all nodes attempt to download the repository.

Usage: python download_hf_hub_repo.py
"""
from huggingface_hub import snapshot_download

# Fill in these args here.
# ex. NousResearch/Llama-2-7b-hf
REPO_ID = "<repo_id>"
# ex. /fsx/users/viczhu/third_party/huggingface/models/NousResearch/Llama-2-7b-hf/
LOCAL_DIR = "<fsx_local_dir_path>"
# End fill in args.

downloaded_repo_path = snapshot_download(
    repo_id=REPO_ID,
    local_dir=LOCAL_DIR,
    force_download=True,
    local_dir_use_symlinks=False,
)
print("Downloaded repository path:", downloaded_repo_path)
