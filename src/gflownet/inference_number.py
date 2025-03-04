import torch
import torch.optim as optim
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from peft import PeftModel, LoraConfig
from sentence_transformers import SentenceTransformer
import numpy as np
import csv
from typing import List, Dict, Tuple, Optional, Union
import os
import logging
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import math
from dataclasses import dataclass, field
import argparse
from KL_penalty import compute_kl_penalty

# Local imports
from model import get_lora_model
from better_generate import generate_sequences_with_logits
from interceptor import RawTextProcessor
from state import Concept, ConceptBlock, OpenBlock, AbstractBlock

# Define dataclasses for parameter groups
@dataclass
class ModelConfig:
    """Configuration for model and tokenizer."""
    model: PeftModel
    tokenizer: AutoTokenizer
    text_processor: RawTextProcessor
    sentence_transformer: Optional[SentenceTransformer] = None
    reference_model: Optional[PeftModel] = None

@dataclass
class TextGenerationConfig:
    """Configuration for text generation."""
    prompt: str
    max_new_tokens: int = 40
    batch_size: int = 16
    generation_config: Optional[GenerationConfig] = None

@dataclass
class DiversityConfig:
    """Configuration for diversity training."""
    n_clusters: int = 5
    max_n_clusters: int = 10
    fixed_n_clusters: bool = False
    w_c: float = 0.5  # Weight for concept loss
    w_o: float = 0.5  # Weight for open block loss
    w_kl: float = 0.1  # Weight for KL penalty
    num_samples: int = 100  # Number of samples for initializing replay buffer
    buffer_size: int = 500  # Maximum size of the replay buffer
    update_clusters_every: int = 20  # Update clusters every N samples
    min_samples_for_clustering: int = 10  # Minimum samples required for clustering

@dataclass
class TrainingConfig:
    """Configuration for training process."""
    num_epochs: int = 10
    num_steps_per_epoch: int = 10
    learning_rate: float = 5e-5
    final_learning_rate: float = 1e-6
    warmup_steps: int = 0
    lr_scheduler_type: str = "cosine"
    output_dir: str = "diversity_model"

@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging."""
    use_wandb: bool = False
    wandb_project: str = "gfn-diversity"
    wandb_name: Optional[str] = None

def extract_numbers_from_concepts(sequences, tokenizer, text_processor):
    """
    Extract numbers from ConceptBlocks in generated sequences.
    """
    extracted_numbers = []
    for sequence in sequences:
        decoded_texts = [tokenizer.decode(token, skip_special_tokens=True) for token in sequence]
        trajectory, _ = text_processor.process_text_to_trajectory(decoded_texts)

        # Collect only the numbers (options) from ConceptBlocks
        numbers = [block.option for block in trajectory if isinstance(block, ConceptBlock)]
        extracted_numbers.append(",".join(numbers))  # Convert list to comma-separated string
    return extracted_numbers

def inference(
    model_config: ModelConfig,
    gen_config: TextGenerationConfig,
) -> None:
    """
    Train the model to generate diverse responses.
    
    Args:
        model_config: Configuration for model and tokenizer
        gen_config: Configuration for text generation
        diversity_config: Configuration for diversity training
        training_config: Configuration for training process
        wandb_config: Configuration for Weights & Biases logging (optional)
    """

    with torch.no_grad():
        generations = generate_sequences_with_logits(
            prompt=gen_config.prompt,
            model=model_config.model,
            tokenizer=model_config.tokenizer,
            batch_size=gen_config.batch_size,
            max_new_tokens=gen_config.max_new_tokens,
            generation_config=gen_config.generation_config
        )
        
        extracted_numbers = extract_numbers_from_concepts(generations["sequences"], model_config.tokenizer, model_config.text_processor)
        return extracted_numbers
        

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a model for diverse text generation")
    
    # Model and tokenizer arguments
    MODEL_PATH_DEFAULT = "/net/scratch2/listar2000/gfn-od/models/pretrained/Meta-Llama-3-8B-Instruct"
    PROMPT_DEFAULT = "Generate 5 flower names, separated by commas. Answer:"
    BASE_OUTPUT_DIR="/net/scratch/listar2000/gfn-od/models/finetuned/train_animal"

    parser = argparse.ArgumentParser(description="Run inference on fine-tuned models")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the fine-tuned LoRA adapter")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--prompt", type=str, default="Generate 6 random numbers from 1 to 6, separated by commas:", help="Input prompt for generation")
    parser.add_argument("--N", type=int, default=32, help="Total number of inferences to run")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--output_csv_base", type=str, default="inference_base.csv", help="CSV file for base model results")
    parser.add_argument("--output_csv_adapter", type=str, default="inference_finetuned.csv", help="CSV file for fine-tuned model results")
    parser.add_argument("--max_new_tokens", type=int, default=30, help="Maximum number of new tokens to generate")
    parser.add_argument("--use_base", type=bool, default=True, help="Path to the fine-tuned LoRA adapter")
    parser.add_argument(
        "--model_type",
        choices=["base", "adapter"],
        default="base",
        help="Specify the model type: 'base' (default) or 'adapter'."
    )
    args = parser.parse_args()
    
    
    sentence_transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder="/net/scratch/jiaweizhang")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        local_files_only=True,
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id
    """
    Run inference N times and save extracted numbers to CSV.
    """
    N = args.N
    if args.model_type == "base":
        output_csv = args.output_csv_base
        model_type = "base"
    else:  # Adapter case
        output_csv = args.output_csv_adapter
        model_type = "fine-tuned"
        model = PeftModel.from_pretrained(model, args.adapter_path)

    
    ListOfNumbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    # Create text processor with N flower concept
    N_Concepts = 3
    ConceptNames= [Concept(f"Number{i+1}", ListOfNumbers, case_variants=["capitalized", "lower", "plural"]) for i in range(N_Concepts)]
    text_processor = RawTextProcessor(ConceptNames, max_window_size=N_Concepts, only_concepts=True)
    """
    #[Red, Blue, Green, Yellow, Orange, Purple]
    ListOfColors = [ 'Red', 'Blue', 'Green', 'Yellow', 'Orange', 'Purple']
    N_Concepts = 3
    ConceptNames = [Concept(f"Color{i+1}", ListOfColors, case_variants=["capitalized", "lower", "plural"]) for i in range(N_Concepts)]
    text_processor = RawTextProcessor(ConceptNames, max_window_size=N_Concepts, only_concepts=True)
    """
    # Set up generation config
    generation_config = GenerationConfig(
        temperature=1.0,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        stop_strings=["\n", ".\n\n", ".\n"]
    )
    
    # Create configuration objects
    model_config = ModelConfig(
        model=model, 
        tokenizer=tokenizer, 
        text_processor=text_processor, 
        sentence_transformer=sentence_transformer,
        reference_model=None
    )
    
    gen_config = TextGenerationConfig(
        prompt=args.prompt, 
        batch_size=args.batch_size, 
        max_new_tokens=args.max_new_tokens, 
        generation_config=generation_config
    )

    results = []
    for i in range(N // gen_config.batch_size):  # Loop through in batches
        print(f"\nRunning batch {i + 1}/{N // gen_config.batch_size} for {model_type} model...")

        extracted_numbers = inference(model_config, gen_config)

        for j in range(gen_config.batch_size):
            results.append([i * gen_config.batch_size + j + 1, model_type, extracted_numbers[j]])

    # Save results to CSV
    with open(output_csv, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Run", "Model Type", "Extracted Numbers"])
        writer.writerows(results)

    print(f"\nInference completed for {model_type}. Results saved to {output_csv}")
        