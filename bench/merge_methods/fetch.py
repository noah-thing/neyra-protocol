"""Download public checkpoints for the merge study into ~/merge-exp/models. usage: mm_fetch.py NAME [NAME ...] | rm NAME ..."""

import os
import shutil
import sys

from huggingface_hub import snapshot_download

ROOT = os.environ.get("MM_ROOT", os.path.expanduser("~/merge-exp")) + "/models"
REPOS = {
    "base": "Qwen/Qwen2.5-0.5B",
    "coder": "Qwen/Qwen2.5-Coder-0.5B",
    "instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "grpo": "jaygala24/Qwen2.5-0.5B-GRPO-math-reasoning",
    "dapo": "jaygala24/Qwen2.5-0.5B-DAPO-math-reasoning",
    "rloo": "jaygala24/Qwen2.5-0.5B-RLOO-math-reasoning",
    "dolphin": "dphn/Dolphin3.0-Qwen2.5-0.5B",
}

if __name__ == "__main__":
    if sys.argv[1] == "rm":
        for n in sys.argv[2:]:
            shutil.rmtree(f"{ROOT}/{n}", ignore_errors=True)
        print("removed", sys.argv[2:])
    else:
        for n in sys.argv[1:]:
            p = snapshot_download(REPOS[n], local_dir=f"{ROOT}/{n}", allow_patterns=["*.safetensors", "*.json", "*.txt"])
            print(n, p, sorted(os.listdir(p)), flush=True)
