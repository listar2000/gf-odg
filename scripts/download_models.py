#!/usr/bin/env python3
"""
Utility script to download and cache models from HuggingFace.
Handles authentication and ensures models are stored in the correct location.
"""

import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import inquirer
import torch

load_dotenv()

# Get the project root by going up one level from the script location
PROJECT_ROOT = Path(__file__).resolve().parent.parent

AVAILABLE_MODELS = [
    "gemma-2b",
    "llama3-8b",
    "phi4"  # Add more models here
]

def get_model_config(model_name: str) -> DictConfig:
    """Load model configuration."""
    config_path = PROJECT_ROOT / "configs/models" / f"{model_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)

def list_available_models() -> list[dict]:
    """List available models with their display names."""
    models = []
    for model_name in AVAILABLE_MODELS:
        try:
            config = get_model_config(model_name)
            models.append({
                'name': model_name,
                'display': config.model.display_name
            })
        except Exception as e:
            print(f"Warning: Could not load config for {model_name}: {e}")
    return models

def check_hf_token():
    """Check if HF_TOKEN is set and valid."""
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN environment variable not set!")
        print("Please set it in your .env file or environment.")
        print("You can get your token from: https://huggingface.co/settings/tokens")
        return False
    return True

def get_model_save_dir(model_name: str, pretrained_dir: str) -> Path:
    """Get the directory where the model should be saved.
    Uses the last part of the model name (after the last /) as the directory name.
    """
    model_dir_name = model_name.split("/")[-1]
    return Path(pretrained_dir) / model_dir_name

@hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Download and cache the specified model and tokenizer."""
    if not check_hf_token():
        return
        
    # Let user select model
    models = list_available_models()
    if not models:
        print("No valid model configurations found")
        return
        
    questions = [
        inquirer.List('model',
                     message="Which model would you like to download?",
                     choices=[f"{m['display']} ({m['name']})" for m in models],
                     carousel=True)
    ]
    answers = inquirer.prompt(questions)
    if not answers:
        print("No model selected")
        return
    
    # Extract model name from selection
    selected = answers['model']
    model_name = selected.split('(')[-1].rstrip(')')
    
    # Load model config
    model_cfg = get_model_config(model_name).model
    print(f"Downloading model: {model_cfg.name}")
    
    # Create model directory in pretrained/
    model_dir = get_model_save_dir(model_cfg.name, cfg.paths.pretrained)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.name,
            token=model_cfg.use_auth_token  # Using token instead of use_auth_token
        )
        
        # Download model
        print("Downloading model...")
        # Convert string dtype to torch dtype
        torch_dtype = getattr(torch, model_cfg.torch_dtype) if hasattr(torch, model_cfg.torch_dtype) else None
        
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg.name,
            token=model_cfg.use_auth_token,  # Using token instead of use_auth_token
            torch_dtype=torch_dtype,
            **{k: v for k, v in model_cfg.model_kwargs.items() if k != 'torch_dtype'}  # Remove torch_dtype from kwargs
        )
        
        # Save to pretrained directory
        print(f"Saving to {model_dir}")
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
        
        print("Download complete!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nPossible solutions:")
        print("1. Check if your HF_TOKEN is correct")
        print("2. Make sure you have access to the model (some models require approval)")
        print("3. Verify the model name in the config file is correct")
        print("\nFor more help, visit: https://huggingface.co/docs/hub/security-tokens")

if __name__ == "__main__":
    main()
