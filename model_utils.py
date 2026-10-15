"""Shared, offline loading of the Qwen model already cached on your Mac."""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_DIR = Path.home() / "jspace-data"


def load_model():
    """Return (model, tokenizer, device); importing this file loads nothing."""
    cache = str(DATA_DIR / "models")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=cache, local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, cache_dir=cache, local_files_only=True,
        torch_dtype=torch.float32,
    )
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer, device
