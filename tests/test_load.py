#!/usr/bin/env python3
"""Test loading the model from local directory"""

import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path

@hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Load model config - Hydra will handle all path interpolation
    model_path = Path(cfg.paths.pretrained) / "Meta-Llama-3-8B-Instruct"
    print("Loading model from:", model_path)
    
    # Try loading tokenizer and model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True  # Force local loading
    )
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True  # Force local loading
    )
    
    print("Successfully loaded model and tokenizer!")
    
if __name__ == "__main__":
    main()
