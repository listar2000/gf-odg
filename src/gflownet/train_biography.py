from train_animal import (
    ModelConfig, TextGenerationConfig, TrainingConfig, WandbConfig, DiversityConfig,
    calculate_concept_kl, calculate_open_block_kl, fill_blocks_with_probs
)
from interceptor import RawTextProcessor
from state import Concept, AbstractBlock, OpenBlock, ConceptBlock
from typing import List, Tuple, Dict, Optional
import argparse, logging
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from state import ConceptBlock, OpenBlock
from replay_buffer import DiversityReplayBuffer
from model import get_lora_model
from transformers import GenerationConfig
from better_generate import generate_sequences_with_logits
import wandb, os, math
from tqdm import tqdm


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


def extract_state_party_blocks_from_trajectories(
    trajectories: List[List[AbstractBlock]],
) -> Tuple[List[ConceptBlock], Dict[str, List[OpenBlock]]]:
    """
    Extract concept blocks and open blocks from trajectories.
    For concept blocks, only extract those with the specified concept name.
    For open blocks, extract those that follow the specified concept.
    """
    state_blocks = [] # for storing blocks for different U.S. states
    party_blocks = [] # for storing blocks for different political parties
    open_block_dict = {}
    
    for trajectory in trajectories:
        prev_party_option = None
        for i, block in enumerate(trajectory):
            if isinstance(block, ConceptBlock) and block.concept.name == "state":
                state_blocks.append(block)
            elif isinstance(block, ConceptBlock) and block.concept.name == "party":
                party_blocks.append(block)
                prev_party_option = block.option
            elif isinstance(block, OpenBlock) and prev_party_option is not None and i > 0:
                # This open block follows a concept block of interest
                if isinstance(trajectory[i-1], ConceptBlock) and trajectory[i-1].concept.name == "state":
                    if open_block_dict.get(prev_party_option) is None:
                        open_block_dict[prev_party_option] = []
                    open_block_dict[prev_party_option].append(block)
        
    return state_blocks, party_blocks, open_block_dict


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
        max_n_clusters=diversity_config.max_n_clusters,
        fixed_n_clusters=diversity_config.fixed_n_clusters,
        buffer_size=diversity_config.buffer_size,
        update_clusters_every=diversity_config.update_clusters_every,
        min_samples_for_clustering=diversity_config.min_samples_for_clustering
    )
    
    # Generate samples in batches
    initialize_bs = 64
    num_batches = (diversity_config.num_samples + initialize_bs - 1) // initialize_bs
    for _ in tqdm(range(num_batches), desc="Generating samples for replay buffer"):
        # Generate sequences
        generations = generate_sequences_with_logits(
            prompt=gen_config.prompt,
            model=model_config.model,
            tokenizer=model_config.tokenizer,
            batch_size=initialize_bs,
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
        _, _, open_block_dict = extract_state_party_blocks_from_trajectories(trajectories)

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
    
    effective_bs = 0
    all_state_blocks, all_party_blocks, all_open_block_dict = [], [], {}
    while effective_bs < gen_config.batch_size:
        # Generate sequences
        generations = generate_sequences_with_logits(
            prompt=gen_config.prompt,
            model=model_config.model,
            tokenizer=model_config.tokenizer,
            batch_size=gen_config.batch_size_step,
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
        state_blocks, party_blocks, open_block_dict = extract_state_party_blocks_from_trajectories(trajectories)
        all_state_blocks.extend(state_blocks)
        all_party_blocks.extend(party_blocks)
        
        if not all_open_block_dict:
            all_open_block_dict = open_block_dict
        else:
            for concept_option in open_block_dict:
                if concept_option not in all_open_block_dict:
                    all_open_block_dict[concept_option] = []
                all_open_block_dict[concept_option].extend(open_block_dict[concept_option])

        effective_bs = sum([len(block_list) for block_list in all_open_block_dict.values()])

    # naming back
    state_blocks, party_blocks, open_block_dict = all_state_blocks, all_party_blocks, all_open_block_dict
    # Calculate concept loss and open block loss
    concept_loss = calculate_concept_kl(state_blocks, n = 4) + calculate_concept_kl(party_blocks, n = 2)

    open_block_loss = []
    for concept_option in open_block_dict:
        open_blocks = open_block_dict[concept_option]
        texts = ["".join(block.raw_text).strip().replace("{", "").replace("}", "") for block in open_blocks]       
        labels = replay_buffer.add_samples(concept_option=concept_option, texts=texts)
        assert labels is not None, "Labels must not be None"
        n_clusters = replay_buffer.get_n_clusters(concept_option)
        open_block_loss.append(calculate_open_block_kl(open_blocks, labels, n_clusters=n_clusters))
    
    open_block_loss = sum(open_block_loss)
    # Combine losses
    total_loss = diversity_config.w_c * concept_loss + diversity_config.w_o * open_block_loss
    
    # Backpropagate
    total_loss.backward()
    optimizer.step()
    
    return concept_loss.item(), open_block_loss.item(), total_loss.item(), effective_bs


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
            "batch_size_step": gen_config.batch_size_step,  
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
            concept_loss, open_block_loss, total_loss, effective_bs = train_step(
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
                "step": step + 1,
                "effective_batch_size": effective_bs
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
                        batch_size=8,
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
        
        # Save model checkpoint every 5 epochs or at the end of the epoch
        if (epoch + 1) % 5 == 0 or epoch == (training_config.num_epochs - 1):
            checkpoint_dir = os.path.join(training_config.output_dir, f"checkpoint-epoch-{epoch+1}")
            model_config.model.save_pretrained(checkpoint_dir)
            model_config.tokenizer.save_pretrained(checkpoint_dir)
    
    logger.info(f"Training completed. Model saved to {training_config.output_dir}")
    
    # Finish wandb run if enabled
    if wandb_config and wandb_config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a model for diverse text generation")
    
    # Model and tokenizer arguments
    MODEL_PATH_DEFAULT = "/net/scratch2/listar2000/gfn-od/models/pretrained/Meta-Llama-3-8B-Instruct"
    PROMPT_DEFAULT ="Generate a synthetic profile of a US politician. Your response must follow this exact structured format: {Party}{Home State}{Political Agenda} Where: Party: choose exactly one from Democrat or Republican. Home State: Choose exactly one from California, Texas, Michigan, Illinois. Political Agenda: Write a concise, one-sentence description of the politician’s main political agenda. Some examples: {Republican}{Alabama}{Less taxation for business}, {Democrat}{Washington}{Climate change}; Answer {"
    BASE_OUTPUT_DIR="/net/scratch2/listar2000/gfn-od/models/finetuned/train_biography"

    parser.add_argument("--model_name_or_path", type=str, default=MODEL_PATH_DEFAULT, help="Model name or path")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    
    # Text generation arguments
    parser.add_argument("--prompt", type=str, default=PROMPT_DEFAULT, help="Prompt for text generation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--batch_size_step", type=int, default=8, help="Actual batch size for training")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Maximum number of new tokens to generate")
    
    # Diversity config arguments
    parser.add_argument("--concept_name", type=str, default="biography", help="Name of the concept to diversify")
    parser.add_argument("--n_clusters", type=int, default=2, help="Number of clusters for diversity")
    parser.add_argument("--max_n_clusters", type=int, default=10, help="Maximum number of clusters for diversity")
    parser.add_argument("--fixed_n_clusters", type=bool, default=False, help="Whether to fix the number of clusters")
    parser.add_argument("--num_samples", type=int, default=320, help="Number of samples for initializing replay buffer")
    
    parser.add_argument("--w_c", type=float, default=0.5, help="Weight for concept loss")
    parser.add_argument("--w_o", type=float, default=0.5, help="Weight for open block loss")
    parser.add_argument("--buffer_size", type=int, default=500, help="Maximum size of the replay buffer")
    parser.add_argument("--update_clusters_every", type=int, default=100, help="Update clusters every N samples")
    parser.add_argument("--min_samples_for_clustering", type=int, default=20, help="Minimum samples required for clustering")
    
    # Training config arguments
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--num_steps_per_epoch", type=int, default=20, help="Number of steps per epoch")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate")
    parser.add_argument("--final_learning_rate", type=float, default=3e-5, help="Final learning rate")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["cosine", "linear", "constant"], help="Type of learning rate scheduler")
    parser.add_argument("--output_dir", type=str, default=BASE_OUTPUT_DIR + "/tmp", help="Directory to save the model")
    
    # Wandb config arguments
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="gfn-diversity", help="Weights & Biases project name")
    parser.add_argument("--wandb_name", type=str, default="train_biography", help="Weights & Biases run name")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    
    # Load model and tokenizer
    model, tokenizer, sentence_transformer = get_lora_model(
        model_name_or_path=args.model_name_or_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    # Create text processor with the two concepts, party and state
    party = Concept("party", ["democrat", "republican"], case_variants=["capitalized", "upper", "plural"])
    state = Concept("state", ["california", "michigan", "texas", "illinois"], case_variants=["capitalized", "upper", "plural"])
    text_processor = RawTextProcessor([party, state], max_window_size=2)
    
    # Set up generation config
    generation_config = GenerationConfig(
        temperature=1.0,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        stop_strings=["\n", ".\n\n", ".\n", "."]
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
        batch_size_step=args.batch_size_step,
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

    # with torch.no_grad():
    #     # Generate sequences
    #     generations = generate_sequences_with_logits(
    #         prompt=gen_config.prompt,
    #         model=model_config.model,
    #         tokenizer=model_config.tokenizer,
    #         batch_size=gen_config.batch_size,
    #         max_new_tokens=gen_config.max_new_tokens,
    #         generation_config=gen_config.generation_config
    #     )
        
    #     # Process sequences into trajectories
    #     trajectories, idxs = [], []
    #     for sequence in generations["sequences"]:
    #         decoded_list = [model_config.tokenizer.decode(token, skip_special_tokens=True) for token in sequence]
    #         trajectory, idx = model_config.text_processor.process_text_to_trajectory(decoded_list)
    #         # print(trajectory, idx)
    #         trajectories.append(trajectory)
    #         idxs.append(idx)

    #     fill_blocks_with_probs(trajectories, idxs, generations["probabilities"])
        
    #     # Extract concept blocks and open blocks
    #     state_blocks, party_blocks, open_block_dict = extract_state_party_blocks_from_trajectories(trajectories)