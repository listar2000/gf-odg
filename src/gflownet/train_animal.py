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

# Local imports
from model import get_lora_model
from better_generate import generate_sequences_with_logits
from interceptor import RawTextProcessor
from state import Concept, ConceptBlock, OpenBlock, AbstractBlock
from replay_buffer import DiversityReplayBuffer

# Define dataclasses for parameter groups
@dataclass
class ModelConfig:
    """Configuration for model and tokenizer."""
    model: PeftModel
    tokenizer: AutoTokenizer
    text_processor: RawTextProcessor
    sentence_transformer: Optional[SentenceTransformer] = None

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
    w_c: float = 0.5  # Weight for concept loss
    w_o: float = 0.5  # Weight for open block loss
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


def calculate_concept_kl(
    concept_blocks: List[ConceptBlock]
) -> torch.Tensor:
    """
    Calculate the KL divergence between the empirical distribution and a uniform distribution, where the 
    empirical distribution is the distribution of concept options.
    """
    # Group blocks by option
    option_blocks = {}
    for block in concept_blocks:
        option = block.option
        if option not in option_blocks:
            option_blocks[option] = []
        option_blocks[option].append(block)
    
    # Sum probabilities for each option while maintaining gradients
    empirical_probs = []
    for option, blocks in option_blocks.items():
        # Sum the probabilities for this option
        option_prob = sum([block.prob for block in blocks])
        empirical_probs.append(option_prob)
    
    # Stack into a tensor to maintain gradient flow
    empirical_probs = torch.stack(empirical_probs)
    
    # Check if sum is positive
    assert empirical_probs.sum() > 0, "Total probability must be > 0"
    assert empirical_probs.requires_grad, "Empirical probs must require gradient"

    # Normalize
    empirical_probs = empirical_probs / empirical_probs.sum()

    uniform_probs = torch.ones_like(empirical_probs) / len(option_blocks)
    return kl_divergence(empirical_probs, uniform_probs, is_log=False)


def calculate_open_block_kl(
    open_blocks: List[OpenBlock],
    labels: np.ndarray,
    n_clusters: int,
) -> torch.Tensor:
    """
    Calculate the open block loss as described in the README.
    L_o = sum_{i=1}^K P[semantic class i] * KL(P[semantic class i] || U)
    where U is the uniform distribution.
    """
    assert len(open_blocks) == len(labels), "Number of open blocks and labels must be the same"
    
    # Get device and dtype from the first block for consistency
    device = open_blocks[0].prob.device
    dtype = open_blocks[0].prob.dtype
    
    # Group blocks by cluster label
    cluster_blocks = {i: [] for i in range(n_clusters)}
    for i, label in enumerate(labels):
        # length normalize the log prob from open_block
        log_prob = torch.log(open_blocks[i].prob).mean()
        # turn log prob into a proper prob
        prob = torch.exp(log_prob)
        cluster_blocks[label].append(prob)
    
    # Create a tensor of zeros with gradient tracking
    empirical_probs = []
    
    # For each cluster, sum the probabilities and add to the corresponding index
    for i in range(n_clusters):
        if cluster_blocks[i]:
            empirical_probs.append(sum(cluster_blocks[i]))
        else:
            empirical_probs.append(torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True))

    empirical_probs = torch.stack(empirical_probs)
    
    # Check if sum is positive
    assert empirical_probs.sum() > 0, "Total probability must be > 0"
    assert empirical_probs.requires_grad, "Empirical probs must require gradient"

    # Normalize
    empirical_probs = empirical_probs / empirical_probs.sum()

    uniform_probs = torch.ones_like(empirical_probs) / n_clusters
    return kl_divergence(empirical_probs, uniform_probs, is_log=False)


def fill_blocks_with_probs(
    trajectories: List[List[AbstractBlock]],
    idxs: List[List[int]],
    raw_probs: List[torch.Tensor]
) -> None:
    """
    Fill blocks with probabilities from the raw probabilities.
    This modifies the blocks in-place.
    """
    for i, (trajectory, idx, prob) in enumerate(zip(trajectories, idxs, raw_probs)):
        for j in range(len(idx)):
            if j < len(idx) - 1:
                start, end = idx[j], idx[j + 1]
            else:
                start, end = idx[j], len(prob)
            
            if j < len(trajectory):
                block = trajectory[j]
                if isinstance(block, ConceptBlock) or isinstance(block, OpenBlock):
                    # For ConceptBlock, we use the product of probabilities
                    if isinstance(block, ConceptBlock):
                        block.prob = torch.prod(prob[start:end])
                    # For OpenBlock, we store the raw probabilities
                    else:
                        block.prob = prob[start:end]


@torch.no_grad()
def initialize_replay_buffer(
    model_config: ModelConfig,
    gen_config: TextGenerationConfig,
    diversity_config: DiversityConfig
) -> DiversityReplayBuffer:
    """
    Initialize the replay buffer with samples from the model.
    
    Args:
        model_config: Configuration for model and tokenizer
        gen_config: Configuration for text generation
        diversity_config: Configuration for diversity training
            - concept_name: Name of the concept to diversify
            - n_clusters: Number of clusters for diversity
            - num_samples: Number of samples to generate
            - buffer_size: Maximum size of the replay buffer
            - update_clusters_every: Update clusters every N samples
            - min_samples_for_clustering: Minimum samples required for clustering
    """
    logging.info(f"Initializing replay buffer with {diversity_config.num_samples} samples")
    
    # Create the replay buffer
    replay_buffer = DiversityReplayBuffer(
        embedder=model_config.sentence_transformer,
        n_clusters=diversity_config.n_clusters,
        buffer_size=diversity_config.buffer_size,
        update_clusters_every=diversity_config.update_clusters_every,
        min_samples_for_clustering=diversity_config.min_samples_for_clustering
    )
    
    # Generate samples in batches
    num_batches = (diversity_config.num_samples + gen_config.batch_size - 1) // gen_config.batch_size
    for _ in tqdm(range(num_batches), desc="Generating samples for replay buffer"):
        # Generate sequences
        generations = generate_sequences_with_logits(
            prompt=gen_config.prompt,
            model=model_config.model,
            tokenizer=model_config.tokenizer,
            batch_size=gen_config.batch_size,
            max_new_tokens=gen_config.max_new_tokens,
            generation_config=gen_config.generation_config
        )
        
        # Process sequences into trajectories
        trajectories = []
        for sequence in generations["sequences"]:
            decoded_list = [model_config.tokenizer.decode(token, skip_special_tokens=True) for token in sequence]
            trajectory, idx = model_config.text_processor.process_text_to_trajectory(decoded_list)
            # print(trajectory, idx)
            trajectories.append(trajectory)
        
        # Extract concept blocks and open blocks
        _, open_block_dict = extract_blocks_from_trajectories(trajectories, diversity_config.concept_name)

        # Add samples to replay buffer
        for concept_option in open_block_dict:
            texts = ["".join(block.raw_text) for block in open_block_dict[concept_option]] 
            # Prefilling, no need to cluster now.      
            replay_buffer.add_samples(concept_option=concept_option, texts=texts, prefill=True)
        
        diversity_config.num_samples -= gen_config.batch_size
        if diversity_config.num_samples <= 0:
            break
    
    # Log buffer statistics
    stats = replay_buffer.get_stats()
    logging.info(f"Replay buffer initialized with statistics: {stats}")
    
    return replay_buffer


def train_step(
    model_config: ModelConfig,
    gen_config: TextGenerationConfig,
    diversity_config: DiversityConfig,
    optimizer: torch.optim.Optimizer,
    replay_buffer: DiversityReplayBuffer
) -> Tuple[float, float, float]:
    """
    Perform a single training step.
    
    Args:
        model_config: Configuration for model and tokenizer
        gen_config: Configuration for text generation
        diversity_config: Configuration for diversity training
        optimizer: Optimizer for model parameters
        replay_buffer: Replay buffer for diversity training
        
    Returns:
        Tuple of (concept_loss, open_block_loss, total_loss) as float values
    """
    model_config.model.train()
    optimizer.zero_grad()
    
    # Generate sequences
    generations = generate_sequences_with_logits(
        prompt=gen_config.prompt,
        model=model_config.model,
        tokenizer=model_config.tokenizer,
        batch_size=gen_config.batch_size,
        max_new_tokens=gen_config.max_new_tokens,
        generation_config=gen_config.generation_config
    )
    
    # Process sequences into trajectories
    trajectories, idxs = [], []
    for sequence in generations["sequences"]:
        decoded_list = [model_config.tokenizer.decode(token, skip_special_tokens=True) for token in sequence]
        trajectory, idx = model_config.text_processor.process_text_to_trajectory(decoded_list)
        trajectories.append(trajectory)
        idxs.append(idx)

    # Fill probabilities 
    fill_blocks_with_probs(trajectories, idxs, generations["probabilities"])

    # Extract concept blocks and open blocks
    concept_blocks, open_block_dict = extract_blocks_from_trajectories(trajectories, diversity_config.concept_name)

    # Calculate concept loss and open block loss
    concept_loss = calculate_concept_kl(concept_blocks)

    open_block_loss = []
    for concept_option in open_block_dict:
        open_blocks = open_block_dict[concept_option]
        texts = ["".join(block.raw_text) for block in open_blocks]       
        labels = replay_buffer.add_samples(concept_option=concept_option, texts=texts)
        assert labels is not None, "Labels must not be None"
        open_block_loss.append(calculate_open_block_kl(open_blocks, labels, n_clusters=diversity_config.n_clusters))
    
    open_block_loss = sum(open_block_loss)
    # Combine losses
    total_loss = diversity_config.w_c * concept_loss + diversity_config.w_o * open_block_loss
    
    # Backpropagate
    total_loss.backward()
    optimizer.step()
    
    return concept_loss.item(), open_block_loss.item(), total_loss.item()


def train(
    model_config: ModelConfig,
    gen_config: TextGenerationConfig,
    diversity_config: DiversityConfig,
    training_config: TrainingConfig,
    wandb_config: Optional[WandbConfig] = None
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
    
    # Initialize replay buffer
    replay_buffer = initialize_replay_buffer(
        model_config=model_config,
        gen_config=gen_config,
        diversity_config=diversity_config
    )
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(training_config.num_epochs):
        epoch_loss = 0.0
        
        for step in range(training_config.num_steps_per_epoch):
            # Perform training step
            concept_loss, open_block_loss, total_loss = train_step(
                model_config=model_config,
                gen_config=gen_config,
                diversity_config=diversity_config,
                optimizer=optimizer,
                replay_buffer=replay_buffer
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
                "concept_loss": concept_loss,
                "open_block_loss": open_block_loss,
                "learning_rate": current_lr,
                "epoch": epoch + 1,
                "step": step + 1
            }
            
            # Log to wandb if enabled
            if wandb_config and wandb_config.use_wandb:
                wandb.log(metrics)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{training_config.num_epochs}, Step {step+1}/{training_config.num_steps_per_epoch}, Loss: {total_loss:.4f}, LR: {current_lr:.7f}")
            logger.info(f"Concept Loss: {concept_loss:.4f}, Open Block Loss: {open_block_loss:.4f}")
            
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
    parser.add_argument("--model_name_or_path", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    
    # Text generation arguments
    parser.add_argument("--prompt", type=str, default="The animal", help="Prompt for text generation")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=30, help="Maximum number of new tokens to generate")
    
    # Diversity config arguments
    parser.add_argument("--concept_name", type=str, default="animal", help="Name of the concept to diversify")
    parser.add_argument("--n_clusters", type=int, default=5, help="Number of clusters for diversity")
    parser.add_argument("--num_samples", type=int, default=320, help="Number of samples for initializing replay buffer")
    parser.add_argument("--w_c", type=float, default=0.8, help="Weight for concept loss")
    parser.add_argument("--w_o", type=float, default=0.2, help="Weight for open block loss")
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
    parser.add_argument("--output_dir", type=str, default="diversity_model", help="Directory to save the model")
    
    # Wandb config arguments
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="gfn-diversity", help="Weights & Biases project name")
    parser.add_argument("--wandb_name", type=str, default="train_animal", help="Weights & Biases run name")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    
    # Load model and tokenizer
    model, tokenizer, text_processor, sentence_transformer = get_lora_model(
        model_name_or_path=args.model_name_or_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
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
        sentence_transformer=sentence_transformer
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
    train(model_config, gen_config, diversity_config, training_config, wandb_config)