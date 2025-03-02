import torch
import torch.optim as optim
from transformers import AutoTokenizer, GenerationConfig
from peft import PeftModel, LoraConfig
from sentence_transformers import SentenceTransformer
import numpy as np
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
    concept_name: str
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


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('diversity_training.log')
    ]
)
logger = logging.getLogger(__name__)

def extract_blocks_from_trajectories(
    trajectories: List[List[AbstractBlock]],
    concept_name: str
) -> Tuple[List[ConceptBlock], Dict[str, List[OpenBlock]]]:
    """
    Extract concept blocks and open blocks from trajectories.
    For concept blocks, only extract those with the specified concept name.
    For open blocks, extract those that follow the specified concept.
    """
    concept_blocks = []
    open_block_dict = {}
    
    for trajectory in trajectories:
        prev_concept_option = None
        for i, block in enumerate(trajectory):
            if isinstance(block, ConceptBlock) and block.concept.name == concept_name:
                concept_blocks.append(block)
                prev_concept_option = block.option
            elif isinstance(block, OpenBlock) and prev_concept_option is not None and i > 0:
                # This open block follows a concept block of interest
                if isinstance(trajectory[i-1], ConceptBlock) and trajectory[i-1].concept.name == concept_name:
                    if open_block_dict.get(prev_concept_option) is None:
                        open_block_dict[prev_concept_option] = []
                    open_block_dict[prev_concept_option].append(block)
    
    return concept_blocks, open_block_dict

def kl_divergence(p: torch.Tensor, q: torch.Tensor, is_log: bool = False):
    """
    Calculate KL divergence between distributions p and q.
    Adds small epsilon to avoid numerical issues with log(0).
    
    if `is_log` is True, then we assume the input tensors to represent log probabilities.
    
    Note: Handles the case where p[i]=0 by using the fact that lim_{p->0} p*log(p/q) = 0
    """
    epsilon = 1e-10  # Small value to avoid numerical issues
    
    if is_log:
        # make sure all the elements are <= 0
        assert (p <= 0).all(), "p must be <= 0"
        assert (q <= 0).all(), "q must be <= 0"
        diffs = p - q  # essentially log(p / q)
        return (torch.exp(p) * diffs).sum()
    else:
        # Normalize to ensure they sum to 1
        p = p / p.sum()
        q = q / q.sum()
        
        # Add epsilon to q to avoid division by zero
        q = q + epsilon
        q = q / q.sum()  # Renormalize after adding epsilon
        
        # Create a mask for p > 0 to handle the case where p[i]=0
        mask = p > 0
        
        # Calculate KL divergence only for non-zero p values
        # For p[i]=0, the contribution is 0 (lim_{p->0} p*log(p/q) = 0)
        kl = torch.zeros_like(p)
        kl[mask] = p[mask] * torch.log(p[mask] / q[mask])
        
        return kl.sum()


def calculate_concept_kl(concept_blocks: List[ConceptBlock], concept: Concept) -> torch.Tensor:
    """
    Calculate the KL divergence between the empirical distribution and a uniform distribution,
    ensuring that all concept options are included in the probability distribution.
    """
    # Detect the device dynamically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize probability dictionary with zero probabilities for all options in the concept
    option_probs = {
        option: torch.tensor(0.0, requires_grad=True, device=device)  # Ensure tensor is on the correct device
        for option in concept.options
    }

    # Sum probabilities for each option
    for block in concept_blocks:
        option = block.option
        if option in option_probs:
            option_probs[option] = option_probs[option] + block.prob.to(device)  # Move block.prob to the same device

    # Convert probabilities to a tensor
    empirical_probs = torch.stack(list(option_probs.values())).to(device)  # Ensure stacking happens on the correct device

    # Ensure the sum of probabilities is positive
    assert empirical_probs.sum() > 0, "Total probability must be > 0"
    assert empirical_probs.requires_grad, "Empirical probs must require gradient"

    # Normalize empirical probabilities
    empirical_probs = empirical_probs / empirical_probs.sum()

    # Define a uniform distribution over all options
    uniform_probs = torch.ones_like(empirical_probs, device=device) / len(concept.options)

    # Compute KL divergence
    return kl_divergence(empirical_probs, uniform_probs, is_log=False)



def fill_blocks_with_probs(
    trajectories: List[List[AbstractBlock]],
    idxs: List[List[Tuple[int, int]]],
    raw_probs: List[torch.Tensor]
) -> None:
    """
    Fill ConceptBlocks with probabilities from the raw probabilities.
    This modifies the blocks in-place.
    """
    for trajectory, idx_pairs, prob in zip(trajectories, idxs, raw_probs):
        for j, (start, end) in enumerate(idx_pairs):
            if j < len(trajectory):
                block = trajectory[j]
                if isinstance(block, ConceptBlock):  
                    # For ConceptBlock, use the product of probabilities
                    block.prob = torch.prod(prob[start:end])

def train_step(
    model_config: ModelConfig,
    gen_config: TextGenerationConfig,
    diversity_config: DiversityConfig,
    optimizer: torch.optim.Optimizer,
    ConceptNames: List[Concept]
) -> Tuple[List[torch.Tensor], float, float]:
    """
    Perform a single training step with KL divergence from a reference model.

    Args:
        model_config: Configuration for model and tokenizer
        gen_config: Configuration for text generation
        diversity_config: Configuration for diversity training
        optimizer: Optimizer for model parameters

    Returns:
        Tuple of (concept_loss_list, kl_penalty, total_loss) as float values.
    """
    model_config.model.train()
    optimizer.zero_grad()

    ### **Step 1: Generate sequences with fine-tuned model** ###
    generations = generate_sequences_with_logits(
        prompt=gen_config.prompt,
        model=model_config.model,
        tokenizer=model_config.tokenizer,
        batch_size=gen_config.batch_size,
        max_new_tokens=gen_config.max_new_tokens,
        generation_config=gen_config.generation_config
    )

    ### **Step 2: Generate sequences with reference model (no gradients)** ###
    with torch.no_grad():
        reference_generations = generate_sequences_with_logits(
            prompt=gen_config.prompt,
            model=model_config.reference_model,
            tokenizer=model_config.tokenizer,
            batch_size=gen_config.batch_size,
            max_new_tokens=gen_config.max_new_tokens,
            generation_config=gen_config.generation_config
        )

    
    ### **Step 3: Compute KL divergence loss using `compute_kl_penalty`** ###
    device = model_config.model.device  # Ensure correct device usage

    # Ensure logits exist before computing KL
    if not generations["logits"] or not reference_generations["logits"]:
        raise ValueError("Logits are missing from model or reference model outputs.")

    # Compute KL penalty using the imported function
    kl_penalty = compute_kl_penalty(generations["logits"], reference_generations["logits"], device=device)

    ### **Step 4: Process sequences into trajectories** ###
    trajectories, idxs = [], []
    for sequence in generations["sequences"]:
        decoded_list = [model_config.tokenizer.decode(token, skip_special_tokens=True) for token in sequence]
        trajectory, idx = model_config.text_processor.process_text_to_trajectory(decoded_list)
        trajectories.append(trajectory)
        idxs.append(idx)

    # Fill probabilities
    fill_blocks_with_probs(trajectories, idxs, generations["probabilities"])

    ### **Step 5: Compute Concept Loss** ###
    concept_loss_list = []
    for concept in ConceptNames:
        concept_blocks, _ = extract_blocks_from_trajectories(trajectories, concept.name)
        concept_loss_list.append(calculate_concept_kl(concept_blocks, concept))

    concept_loss = sum(concept_loss_list)

    ### **Step 6: Compute Total Loss** ###
    total_loss = concept_loss + diversity_config.w_kl * kl_penalty

    ### **Step 7: Backpropagation and Optimization** ###
    total_loss.backward()
    optimizer.step()

    return concept_loss_list, kl_penalty, total_loss.item()



def train(
    model_config: ModelConfig,
    gen_config: TextGenerationConfig,
    diversity_config: DiversityConfig,
    training_config: TrainingConfig,
    ConceptNames: List[Concept],
    wandb_config: Optional[WandbConfig] = None,
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
    # Create output directory
    os.makedirs(training_config.output_dir, exist_ok=True)
    
    # Initialize wandb if requested
    if wandb_config and wandb_config.use_wandb:
        wandb_config_dict = {
            "learning_rate": training_config.learning_rate,
            "final_learning_rate": training_config.final_learning_rate,
            "warmup_steps": training_config.warmup_steps,
            "lr_scheduler_type": training_config.lr_scheduler_type,
            "num_epochs": training_config.num_epochs,
            "num_steps_per_epoch": training_config.num_steps_per_epoch,
            "batch_size": gen_config.batch_size,
            "w_c": diversity_config.w_c,
            "w_o": diversity_config.w_o,
            "w_kl": diversity_config.w_kl,
            "n_clusters": diversity_config.n_clusters,
            "max_new_tokens": gen_config.max_new_tokens,
            "num_samples": diversity_config.num_samples,
            "prompt": gen_config.prompt,
            "concept_name": diversity_config.concept_name
        }
        wandb.init(project=wandb_config.wandb_project, name=wandb_config.wandb_name, config=wandb_config_dict)
    
    # Initialize optimizer
    optimizer = optim.AdamW(model_config.model.parameters(), lr=training_config.learning_rate)
    
    # Initialize learning rate scheduler
    total_steps = training_config.num_epochs * training_config.num_steps_per_epoch
    
    if training_config.lr_scheduler_type == "cosine":
        # Cosine scheduler with optional warmup
        if training_config.warmup_steps > 0:
            # Create a warmup + cosine scheduler
            def lr_lambda(current_step: int):
                if current_step < training_config.warmup_steps:
                    return float(current_step) / float(max(1, training_config.warmup_steps))
                progress = float(current_step - training_config.warmup_steps) / float(max(1, total_steps - training_config.warmup_steps))
                return max(training_config.final_learning_rate / training_config.learning_rate, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
            scheduler = LambdaLR(optimizer, lr_lambda)
        else:
            # Simple cosine scheduler
            scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=total_steps,
                eta_min=training_config.final_learning_rate
            )
    elif training_config.lr_scheduler_type == "linear":
        # Linear scheduler with optional warmup
        def lr_lambda(current_step: int):
            if current_step < training_config.warmup_steps:
                return float(current_step) / float(max(1, training_config.warmup_steps))
            return max(
                training_config.final_learning_rate / training_config.learning_rate,
                float(total_steps - current_step) / float(max(1, total_steps - training_config.warmup_steps))
            )
        
        scheduler = LambdaLR(optimizer, lr_lambda)
    else:
        # Constant learning rate (no scheduler)
        scheduler = None
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(training_config.num_epochs):
        epoch_loss = 0.0
        
        for step in range(training_config.num_steps_per_epoch):
            # Perform training step
            concept_loss_list, kl_penalty, total_loss = train_step(
                model_config=model_config,
                gen_config=gen_config,
                diversity_config=diversity_config,
                optimizer=optimizer,
                ConceptNames=ConceptNames
            )
            
            epoch_loss += total_loss
            
            # Update learning rate scheduler
            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = training_config.learning_rate
            
            # Log metrics
            metrics = {
                "total_loss": total_loss,
                "learning_rate": current_lr,
                "epoch": epoch + 1,
                "step": step + 1,
                "kl_penalty": kl_penalty
            }
            
            # Log to wandb if enabled
            if wandb_config and wandb_config.use_wandb:
                wandb.log(metrics)
            
            # Log individual concept losses separately
            for i, loss in enumerate(concept_loss_list):
                metrics[f"concept_loss_{i}"] = loss.item()  # Ensure it's a scalar before logging

            # Log progress
            concept_loss_str = ", ".join([f"Concept {i}: {loss.item():.4f}" for i, loss in enumerate(concept_loss_list)])
            logger.info(f"Epoch {epoch+1}/{training_config.num_epochs}, Step {step+1}/{training_config.num_steps_per_epoch}, Loss: {total_loss:.4f}, LR: {current_lr:.7f}")
            logger.info(f"Concept Losses: {concept_loss_str}")
            logger.info(f"KL Penalty: {kl_penalty:.4f}")

            # Generate and log sample at the end of epoch
            if step == training_config.num_steps_per_epoch - 1:
                with torch.no_grad():
                    samples = generate_sequences_with_logits(
                        prompt=gen_config.prompt,
                        model=model_config.model,
                        tokenizer=model_config.tokenizer,
                        batch_size=4,
                        max_new_tokens=gen_config.max_new_tokens,
                        generation_config=gen_config.generation_config
                    )
                    
                    logger.info("Sample generations:")
                    sample_texts = []
                    for seq in samples["sequences"]:
                        decoded_text = model_config.tokenizer.decode(seq, skip_special_tokens=True)
                        sample_texts.append(decoded_text)
                        logger.info(decoded_text)
                    
                    # Log samples to wandb if enabled
                    if wandb_config and wandb_config.use_wandb:
                        wandb.log({"samples": wandb.Table(columns=["Sample"], data=[[text] for text in sample_texts])})
        
        # Log epoch results
        avg_epoch_loss = epoch_loss / training_config.num_steps_per_epoch
        logger.info(f"Epoch {epoch+1}/{training_config.num_epochs} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Log epoch metrics to wandb if enabled
        if wandb_config and wandb_config.use_wandb:
            wandb.log({"epoch": epoch + 1, "avg_epoch_loss": avg_epoch_loss})
        
        # Save model checkpoint
        # checkpoint_dir = os.path.join(training_config.output_dir, f"checkpoint-epoch-{epoch+1}")
        # model_config.model.save_pretrained(checkpoint_dir)
        # model_config.tokenizer.save_pretrained(checkpoint_dir)
        
        # Update replay buffer with new embeddings
        # This would involve collecting more samples and updating the embeddings_by_option
        # and cluster_centers_by_option dictionaries
    
    # Save final model
    model_config.model.save_pretrained(training_config.output_dir)
    model_config.tokenizer.save_pretrained(training_config.output_dir)
    logger.info(f"Training completed. Model saved to {training_config.output_dir}")
    
    # Finish wandb run if enabled
    if wandb_config and wandb_config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a model for diverse text generation")
    
    # Model and tokenizer arguments
    MODEL_PATH_DEFAULT = "/net/scratch2/listar2000/gfn-od/models/pretrained/Meta-Llama-3-8B-Instruct"
    PROMPT_DEFAULT = "Generate 5 flower names, separated by commas. Answer:"
    BASE_OUTPUT_DIR="/net/scratch/listar2000/gfn-od/models/finetuned/train_animal"

    parser.add_argument("--model_name_or_path", type=str, default=MODEL_PATH_DEFAULT, help="Model name or path")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    
    # Text generation arguments
    parser.add_argument("--prompt", type=str, default=PROMPT_DEFAULT, help="Prompt for text generation")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=30, help="Maximum number of new tokens to generate")
    
    # Diversity config arguments
    parser.add_argument("--concept_name", type=str, default="animal", help="Name of the concept to diversify")
    parser.add_argument("--n_clusters", type=int, default=5, help="Number of clusters for diversity")
    parser.add_argument("--max_n_clusters", type=int, default=10, help="Maximum number of clusters for diversity")
    parser.add_argument("--fixed_n_clusters", type=bool, default=False, help="Whether to fix the number of clusters")
    parser.add_argument("--num_samples", type=int, default=320, help="Number of samples for initializing replay buffer")
    
    parser.add_argument("--w_c", type=float, default=0.8, help="Weight for concept loss")
    parser.add_argument("--w_o", type=float, default=0.2, help="Weight for open block loss")
    parser.add_argument("--w_kl", type=float, default=0.1, help="Weight for KL penalty")
    parser.add_argument("--buffer_size", type=int, default=500, help="Maximum size of the replay buffer")
    parser.add_argument("--update_clusters_every", type=int, default=100, help="Update clusters every N samples")
    parser.add_argument("--min_samples_for_clustering", type=int, default=20, help="Minimum samples required for clustering")
    
    # Training config arguments
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--num_steps_per_epoch", type=int, default=10, help="Number of steps per epoch")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--final_learning_rate", type=float, default=3e-5, help="Final learning rate")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["cosine", "linear", "constant"], help="Type of learning rate scheduler")
    parser.add_argument("--output_dir", type=str, default=BASE_OUTPUT_DIR + "/tmp", help="Directory to save the model")
    
    # Wandb config arguments
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="gfn-diversity", help="Weights & Biases project name")
    parser.add_argument("--wandb_name", type=str, default="train_flower", help="Weights & Biases run name")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    
    # Load model and tokenizer
    model, tokenizer, sentence_transformer,reference_model = get_lora_model(
        model_name_or_path=args.model_name_or_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    ListOfNumbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    # Create text processor with N flower concept
    N_Concepts = 5
    ConceptNames= [Concept(f"Number{i+1}", ListOfNumbers, case_variants=["capitalized", "lower", "plural"]) for i in range(N_Concepts)]
    text_processor = RawTextProcessor(ConceptNames, max_window_size=N_Concepts, only_concepts=True)
    
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
        reference_model=reference_model
    )
    
    gen_config = TextGenerationConfig(
        prompt=args.prompt, 
        batch_size=args.batch_size, 
        max_new_tokens=args.max_new_tokens, 
        generation_config=generation_config
    )
    
    diversity_config = DiversityConfig(
        concept_name=args.concept_name, 
        n_clusters=args.n_clusters, 
        num_samples=args.num_samples, 
        w_c=args.w_c, 
        w_o=args.w_o, 
        w_kl=args.w_kl,
        buffer_size=args.buffer_size, 
        update_clusters_every=args.update_clusters_every, 
        min_samples_for_clustering=args.min_samples_for_clustering
    )
    
    training_config = TrainingConfig(
        num_epochs=args.num_epochs, 
        num_steps_per_epoch=args.num_steps_per_epoch, 
        learning_rate=args.learning_rate, 
        final_learning_rate=args.final_learning_rate, 
        warmup_steps=args.warmup_steps, 
        lr_scheduler_type=args.lr_scheduler_type, 
        output_dir=args.output_dir
    )
    
    wandb_config = WandbConfig(
        use_wandb=args.use_wandb, 
        wandb_project=args.wandb_project, 
        wandb_name=args.wandb_name
    )
    
    # Train the model
    train(model_config, gen_config, diversity_config, training_config, ConceptNames,wandb_config)