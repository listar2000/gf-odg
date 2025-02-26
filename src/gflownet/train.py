import torch
import torch.optim as optim
from interceptor import RawTextProcessor
from peft import PeftModel, LoraConfig
from model import generate_sequences_with_logits, get_lora_model
from transformers import AutoTokenizer, GenerationConfig
from state import AbstractBlock, Concept, OpenBlock, EndBlock, StartBlock, ConceptBlock
from typing import List
from replay_buffer import calculate_similarity_scores, get_effective_probabilities
from sentence_transformers import SentenceTransformer
from replay_buffer import ReplayBuffer


def get_rewards(trajectories: List[AbstractBlock]) -> torch.Tensor:
    rewards = []
    for traj in trajectories:
        # we require a [start, open, concept, open, end] trajectory 
        # or [start, concept, open, end] trajectory
        # if not isinstance(traj[-2], OpenBlock):
        #     rewards.append(1e-3)
        if not isinstance(traj[-3], ConceptBlock): 
            rewards.append(1e-3)
        else:
            rewards.append(8.0)
            # also check the last open block ends with a "."
            # if not traj[-2].raw_text[-1].strip() == ".":
            #     rewards.append(0.8)
            # else:
            #     rewards.append(1.0)
    return torch.tensor(rewards)


def scale_open_blocks_probs(open_blocks: List[OpenBlock], buffer: ReplayBuffer):
    if not open_blocks:
        return

    # Get text representations of OpenBlocks
    texts = ["".join(block.raw_text) for block in open_blocks]

    # Numerical stability: Convert raw log probabilities
    raw_log_probs = torch.tensor([torch.sum(torch.log(block.prob)) for block in open_blocks], 
                                 dtype=torch.float16, 
                                 device=open_blocks[0].prob.device)

    buffer.update_replay_buffer("open_block", texts)

    probs = buffer.assign_probs(texts, raw_log_probs)

    for block, prob in zip(open_blocks, probs):
        block.prob = prob


def fill_blocks_with_effective_probs(trajectories: List[List[AbstractBlock]], idxs: List[List[int]], \
    raw_probs: List[torch.Tensor], embedder: SentenceTransformer, buffer: ReplayBuffer):
    open_blocks, open_blocks_cat, open_blocks_dog = [], [], []
    for i, idx in enumerate(idxs):
        concept_option = None
        trajectory = trajectories[i][1:-1] # get rid of start and end blocks
        raw_prob = raw_probs[i]
        for j in range(len(idx)):
            if j < len(idx) - 1:
                start, end = idx[j], idx[j + 1]
            else:
                start, end = idx[j], len(raw_prob)
            block = trajectory[j]
            if isinstance(block, ConceptBlock):
                concept_option = block.option
                block.prob = torch.prod(raw_prob[start:end], dim=0)
            elif isinstance(block, OpenBlock):
                block.prob = torch.Tensor(raw_prob[start:end])
                if concept_option is not None:
                    if concept_option == 'cat':
                        open_blocks_cat.append(block)
                    elif concept_option == 'dog':
                        open_blocks_dog.append(block)
                    else:
                        assert False, "Unexpected concept option: " + concept_option
                else:
                    open_blocks.append(block)
    
    # scale_open_blocks_probs(open_blocks, embedder, buffer)
    # scale_open_blocks_probs(open_blocks_cat, embedder, buffer)
    # scale_open_blocks_probs(open_blocks_dog, embedder, buffer)
    return len(open_blocks_cat), len(open_blocks_dog)


def calculate_trajectory_balance_loss(rewards: torch.Tensor, trajectories: List[List[AbstractBlock]], logZ: torch.nn.Parameter):
    losses = []
    rewards = torch.log(rewards)
    for i in range(len(trajectories)):
        trajectory = trajectories[i][1:-1]
        probs = [block.prob for block in trajectory if isinstance(block, ConceptBlock)]
        if len(probs) == 0:
            continue
        else:
            probs = torch.tensor(probs)
        log_probs = torch.sum(torch.log(probs), dim=0)
        losses.append(((logZ + log_probs) - rewards[i]) ** 2)   
    return torch.mean(torch.stack(losses))

def train_single_step(prompt: str, model: PeftModel, tokenizer: AutoTokenizer, \
    logZ: torch.nn.Parameter, optimizer: optim.Optimizer, lr_scheduler: optim.lr_scheduler.LRScheduler, \
    text_processor: RawTextProcessor, embedder: SentenceTransformer, batch_size: int = 64, generation_config: dict = None):
    
    model.train()
    # optimizer.zero_grad()

    generated = generate_sequences_with_logits(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_new_tokens=50,
        sampled_only=True,
        do_padding=False,
        generation_config=generation_config
    )

    # print a few decoded sequences
    print("--------------------------------")
    for seq in generated["sequences"][:2]:
        print(tokenizer.decode(seq, skip_special_tokens=True))

    trajs, idxs = [], []
    for seq in generated["sequences"]:
        decoded = tokenizer.decode(seq, skip_special_tokens=True)
        decoded_list = [tokenizer.decode(token, skip_special_tokens=True) for token in seq]
        traj, idx = text_processor.process_text_to_trajectory(decoded_list)
        trajs.append(traj)
        idxs.append(idx)
        
    rewards = get_rewards(trajs).to(model.device)
    cat_count, dog_count = fill_blocks_with_effective_probs(trajs, idxs, generated["probabilities"], embedder)
    
    loss = calculate_trajectory_balance_loss(rewards, trajs, logZ)

    print(logZ.exp().item())
    print(f"cat count: {cat_count}, dog count: {dog_count}, ratio: {cat_count / dog_count}")
    print("--------------------------------")
    return loss

def replay_buffer_setup(
    buffer: ReplayBuffer, 
    model: PeftModel, 
    tokenizer: AutoTokenizer, 
    text_processor: RawTextProcessor, 
    prompt: str, 
    generation_config: dict
):
    """
    Ensures ReplayBuffer is populated with extracted OpenBlock texts from model-generated output
    before training starts.

    :param buffer: ReplayBuffer instance.
    :param model: The PeftModel used for text generation.
    :param tokenizer: The tokenizer for decoding model-generated sequences.
    :param text_processor: The RawTextProcessor for extracting OpenBlocks.
    :param prompt: The input prompt for text generation.
    :param generation_config: Configuration for generation (temperature, top-p, etc.).
    """
    print("🚀 Collecting initial OpenBlocks for ReplayBuffer using model-generated texts...")

    num_warmup_samples = 16 * 10
    collected_samples = 0

    while collected_samples < num_warmup_samples:
        generated = generate_sequences_with_logits(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            batch_size=16,  
            max_new_tokens=50,
            sampled_only=True,
            do_padding=False,
            generation_config=generation_config
        )

        generated_texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in generated["sequences"]]

        open_texts = []
        for text in generated_texts:
            blocks, _ = text_processor.process_text_to_trajectory(list(text))  # Convert text to list of chars/tokens
            open_texts.extend(["".join(block.raw_text) for block in blocks if isinstance(block, OpenBlock)])

        if open_texts:
            buffer.update_replay_buffer("open_block", open_texts)
            collected_samples += len(open_texts)
            print(f"Collected {collected_samples}/{num_warmup_samples} OpenBlock samples for ReplayBuffer.")


    if "open_block" in buffer.embeddings and len(buffer.embeddings["open_block"]) >= buffer.threshold:
        buffer.train_clusters("open_block")


if __name__ == "__main__": 
    prompt = """In one sentence, explain why do you like cats or dogs (pick only one to answer):
Answer: """
    animal = Concept("animal", ["cat", "dog"], case_variants=["capitalized", "upper", "plural"])
    text_processor = RawTextProcessor([animal], max_window_size=2)
    model_name = "/net/scratch2/listar2000/gfn-od/models/pretrained/Meta-Llama-3-8B-Instruct"
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model, tokenizer = get_lora_model(model_name, lora_config)

    generation_config = GenerationConfig(
        temperature=1.0,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        stop_strings=["\n", ".\n\n"]
    )

    cache_dir = "/net/scratch2/listar2000/gfn-od/models/pretrained/sentence_transformer"
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=cache_dir)
    
    buffer = ReplayBuffer(embedder,n_clusters=2, method="kmeans", threshold=64 * 10, retrain_every=64 * 10)
    replay_buffer_setup(buffer, model, tokenizer, prompt, generation_config)

    logZ = torch.nn.Parameter(torch.zeros(1, dtype=torch.float16, device=model.device), requires_grad=True)
    # optimizer for both model and logZ
    optimizer = optim.AdamW([dict(params=model.parameters(), lr=5e-4), \
        dict(params=logZ, lr=1e-2)])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # print some model parameters
    print("Model parameters:")
    i = 0
    for name, param in model.named_parameters():
        if i > 10:
            break
        i += 1
        print(name, param.requires_grad)

    grad_acc = 4
    for i in range(50):
        optimizer.zero_grad()
        losses = 0.0
        for j in range(grad_acc):
            loss = train_single_step(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                logZ=logZ,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                text_processor=text_processor,
                embedder=embedder,
                batch_size=16,
                generation_config=generation_config
            )
            losses += loss
            loss.backward()
        optimizer.step()
        lr_scheduler.step()
        print("Step:", i)
        print("Loss:", losses.item() / grad_acc)

