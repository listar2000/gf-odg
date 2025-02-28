import torch
import torch.optim as optim
from transformers import AutoTokenizer, GenerationConfig
from peft import PeftModel, LoraConfig
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple
import os
import logging
from tqdm import tqdm

# Local imports
from model import get_lora_model
from better_generate import generate_sequences_with_logits
from interceptor import RawTextProcessor
from state import Concept, ConceptBlock, OpenBlock, AbstractBlock
from new_buffer import DiversityReplayBuffer

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
    prompt: str,
    model: PeftModel,
    tokenizer: AutoTokenizer,
    text_processor: RawTextProcessor,
    sentence_transformer: SentenceTransformer,
    concept_name: str,
    n_clusters: int = 5,
    num_samples: int = 100,
    batch_size: int = 16,
    max_new_tokens: int = 40,
    generation_config: GenerationConfig = None
) -> DiversityReplayBuffer:
    """
    Initialize the replay buffer with samples from the model.
    """
    logging.info(f"Initializing replay buffer with {num_samples} samples")
    
    # Create the replay buffer
    replay_buffer = DiversityReplayBuffer(
        embedder=sentence_transformer,
        n_clusters=n_clusters,
        buffer_size=500,
        update_clusters_every=20,
        min_samples_for_clustering=10
    )
    
    # Generate samples in batches
    num_batches = (num_samples + batch_size - 1) // batch_size
    for _ in tqdm(range(num_batches), desc="Generating samples for replay buffer"):
        # Generate sequences
        generations = generate_sequences_with_logits(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            generation_config=generation_config
        )
        
        # Process sequences into trajectories
        trajectories = []
        for sequence in generations["sequences"]:
            decoded_list = [tokenizer.decode(token, skip_special_tokens=True) for token in sequence]
            trajectory, idx = text_processor.process_text_to_trajectory(decoded_list)
            # print(trajectory, idx)
            trajectories.append(trajectory)
        
        # Extract concept blocks and open blocks
        _, open_block_dict = extract_blocks_from_trajectories(trajectories, concept_name)

        # Add samples to replay buffer
        for concept_option in open_block_dict:
            texts = ["".join(block.raw_text) for block in open_block_dict[concept_option]]       
            replay_buffer.add_samples(concept_option=concept_option, texts=texts)
        
        num_samples -= batch_size
        if num_samples <= 0:
            break
    
    # Log buffer statistics
    stats = replay_buffer.get_stats()
    logging.info(f"Replay buffer initialized with statistics: {stats}")
    
    return replay_buffer


def train_step(
    prompt: str,
    model: PeftModel,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
    text_processor: RawTextProcessor,
    replay_buffer: DiversityReplayBuffer,
    concept_name: str,
    batch_size: int = 16,
    max_new_tokens: int = 40,
    w_c: float = 0.5,
    w_o: float = 0.5,
    n_clusters: int = 5,
    generation_config: GenerationConfig = None
) -> Tuple[float, Dict[str, float], Dict[str, Dict[int, float]]]:
    """
    Perform a single training step.
    """
    model.train()
    optimizer.zero_grad()
    
    # Generate sequences
    generations = generate_sequences_with_logits(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        generation_config=generation_config
    )
    
    # Process sequences into trajectories
    trajectories, idxs = [], []
    for sequence in generations["sequences"]:
        decoded_list = [tokenizer.decode(token, skip_special_tokens=True) for token in sequence]
        trajectory, idx = text_processor.process_text_to_trajectory(decoded_list)
        trajectories.append(trajectory)
        idxs.append(idx)

    # Fill probabilities 
    fill_blocks_with_probs(trajectories, idxs, generations["probabilities"])

    # Extract concept blocks and open blocks
    concept_blocks, open_block_dict = extract_blocks_from_trajectories(trajectories, concept_name)

    # Calculate concept loss and open block loss
    concept_loss = calculate_concept_kl(concept_blocks)

    open_block_loss = []
    for concept_option in open_block_dict:
        open_blocks = open_block_dict[concept_option]
        texts = ["".join(block.raw_text) for block in open_blocks]       
        labels = replay_buffer.add_samples(concept_option=concept_option, texts=texts)
        assert labels is not None, "Labels must not be None"
        open_block_loss.append(calculate_open_block_kl(open_blocks, labels, n_clusters=n_clusters))
    
    open_block_loss = sum(open_block_loss)
    # Combine losses
    total_loss = w_c * concept_loss + w_o * open_block_loss
    
    # Backpropagate
    total_loss.backward()
    optimizer.step()
    
    return concept_loss.item(), open_block_loss.item(), total_loss.item()


def train(
    prompt: str,
    model: PeftModel,
    tokenizer: AutoTokenizer,
    text_processor: RawTextProcessor,
    sentence_transformer: SentenceTransformer,
    concept_name: str,
    n_clusters: int = 5,
    batch_size: int = 16,
    max_new_tokens: int = 40,
    num_epochs: int = 10,
    num_steps_per_epoch: int = 10,
    w_c: float = 0.5,
    w_o: float = 0.5,
    learning_rate: float = 5e-5,
    num_samples: int = 100,
    output_dir: str = "diversity_model",
    generation_config: GenerationConfig = None
) -> None:
    """
    Train the model to generate diverse responses.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Default generation config if not provided
    if generation_config is None:
        generation_config = GenerationConfig(
            temperature=1.0,
            top_p=0.95,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            stop_strings=["\n", ".\n\n", ".\n"]
        )
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Initialize replay buffer
    replay_buffer = initialize_replay_buffer(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        text_processor=text_processor,
        sentence_transformer=sentence_transformer,
        concept_name=concept_name,
        n_clusters=n_clusters,
        num_samples=num_samples,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        generation_config=generation_config
    )
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for step in range(num_steps_per_epoch):
            # Perform training step
            concept_loss, open_block_loss, total_loss = train_step(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                text_processor=text_processor,
                replay_buffer=replay_buffer,
                concept_name=concept_name,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                w_c=w_c,
                w_o=w_o,
                n_clusters=n_clusters,
                generation_config=generation_config
            )
            
            epoch_loss += total_loss
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Step {step+1}/{num_steps_per_epoch}, Loss: {total_loss:.4f}")
            logger.info(f"Concept Loss: {concept_loss:.4f}, Open Block Loss: {open_block_loss:.4f}")
            
            # Generate and log sample at the end of epoch
            if step == num_steps_per_epoch - 1:
                with torch.no_grad():
                    samples = generate_sequences_with_logits(
                        prompt=prompt,
                        model=model,
                        tokenizer=tokenizer,
                        batch_size=4,
                        max_new_tokens=max_new_tokens,
                        generation_config=generation_config
                    )
                    
                    logger.info("Sample generations:")
                    for seq in samples["sequences"]:
                        logger.info(tokenizer.decode(seq, skip_special_tokens=True))
        
        # Log epoch results
        avg_epoch_loss = epoch_loss / num_steps_per_epoch
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Save model checkpoint
        # checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
        # model.save_pretrained(checkpoint_dir)
        # tokenizer.save_pretrained(checkpoint_dir)
        
        # Update replay buffer with new embeddings
        # This would involve collecting more samples and updating the embeddings_by_option
        # and cluster_centers_by_option dictionaries
    
    # Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Training completed. Model saved to {output_dir}")


if __name__ == "__main__":
    # all the paths
    MODEL_DIR = "/net/scratch2/listar2000/gfn-od/models/"
    output_dir = MODEL_DIR + "finetuned/train_animal"
    model_name = MODEL_DIR + "pretrained/Meta-Llama-3-8B-Instruct"
    cache_dir = MODEL_DIR + "pretrained/sentence_transformer"
    
    # Example usage
    prompt = "In 20 words, say whether you love dog or cat more, and give a reason; Answer:"
    
    # Define concept
    animal = Concept("animal", ["cat", "dog"], case_variants=["capitalized", "upper", "lower", "plural"])
    text_processor = RawTextProcessor([animal], max_window_size=2)
    
    # Load model and tokenizer
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model, tokenizer = get_lora_model(model_name, lora_config)
    
    # Load sentence transformer
    sentence_transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=cache_dir)
    
    # Define generation config
    generation_config = GenerationConfig(
        temperature=1.0,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        stop_strings=["\n", ".\n\n", ".\n"]
    )
    
    # Train model
    train(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        text_processor=text_processor,
        sentence_transformer=sentence_transformer,
        concept_name="animal",
        n_clusters=5,
        batch_size=32,
        max_new_tokens=30,
        num_epochs=5,
        num_steps_per_epoch=10,
        w_c=0.8,
        w_o=0.2,
        learning_rate=1e-4,
        num_samples=160,
        output_dir=output_dir,
        generation_config=generation_config
    )