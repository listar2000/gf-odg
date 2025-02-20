from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer
from torch.cuda.amp import autocast

from undecorated import undecorated
from types import MethodType

def generate_sequences_with_logits(
    prompt,
    model,
    tokenizer,
    batch_size=4,
    max_new_tokens=20,
    sampled_only=False,
    do_padding=False,
    generation_config=None
):
    # Create a default generation config if not provided.
    if generation_config is None:
        generation_config = GenerationConfig(
            temperature=1.0,
            top_p=1.0,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            stop_strings=["\n"]
        )
    
    # Tokenize prompt and repeat for batch.
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    input_ids = input_ids.repeat(batch_size, 1)
    attention_mask = attention_mask.repeat(batch_size, 1)
    prompt_length = input_ids.shape[1]
    
    # Set model in train mode (and avoid torch.no_grad) so that gradients are computed.
    model.train()
    model.gradient_checkpointing_enable()
	
    # Lists to collect per-step logits and sampled tokens.
    all_logits = []           # List[Tensor]: each is (batch_size, vocab_size)
    sampled_tokens_list = []  # List[Tensor]: each is (batch_size,)
    
    # Initialize our “current” sequence with the prompt.
    cur_input_ids = input_ids
    cur_attention_mask = attention_mask

    newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[0] 
    newline_end_token_id = tokenizer.encode(".\n\n", add_special_tokens=False)[0]
    # Generation loop (manual autoregressive decoding).
    for step in range(max_new_tokens):
        with autocast():
            # Get the model’s logits for the current sequence.
            outputs = model(cur_input_ids, attention_mask=cur_attention_mask)
            # Next-token logits come from the last position.
            next_logits = outputs.logits[:, -1, :]  # shape: (batch_size, vocab_size)
            all_logits.append(next_logits)
            
            # Scale logits by temperature.
            if generation_config.temperature != 1.0:
                scaled_logits = next_logits / generation_config.temperature
            else:
                scaled_logits = next_logits

            # Top-p (nucleus) filtering if required.
            if generation_config.top_p < 1.0:
                # Sort logits and compute cumulative probabilities.
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above the threshold.
                sorted_indices_to_remove = cumulative_probs > generation_config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                sorted_logits[sorted_indices_to_remove] = -float("inf")
                # Scatter back to original indexing.
                filtered_logits = torch.zeros_like(scaled_logits)
                filtered_logits.scatter_(1, sorted_indices, sorted_logits)
                scaled_logits = filtered_logits

            # Decide the next token.
            if generation_config.do_sample:
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # shape: (batch_size, 1)
            else:
                next_token = scaled_logits.argmax(dim=-1, keepdim=True)
            
            sampled_tokens_list.append(next_token.squeeze(-1))  # store sampled token ids
            
            # Append the new token to each sequence.
            cur_input_ids = torch.cat([cur_input_ids, next_token], dim=1)
            cur_attention_mask = torch.cat(
                [cur_attention_mask, torch.ones((batch_size, 1), dtype=cur_attention_mask.dtype, device=cur_attention_mask.device)],
                dim=1,
            )
            
            # Check stop conditions (EOS or stop-string newline).
            # finished = ((next_token.squeeze(-1) == generation_config.eos_token_id) |
            #             (next_token.squeeze(-1) == newline_token_id) |
            #             (next_token.squeeze(-1) == newline_end_token_id))
            finished = next_token.squeeze(-1) == generation_config.eos_token_id
            if finished.all():
                break

    # Extract the newly generated tokens (i.e. tokens after the prompt).
    generated_new_tokens = cur_input_ids[:, prompt_length:]
    num_steps = generated_new_tokens.shape[1]

    # Stack the per-step logits into one tensor.
    # full_logits: (batch_size, num_steps, vocab_size)
    full_logits = torch.stack(all_logits, dim=1)
    full_probs = torch.softmax(full_logits, dim=-1)

    # If sampled_only, gather the logits/probs corresponding to the sampled token at each step.
    if sampled_only:
        # sampled_tokens: (batch_size, num_steps)
        sampled_tokens = torch.stack(sampled_tokens_list, dim=1)
        sampled_logits = full_logits.gather(dim=2, index=sampled_tokens.unsqueeze(-1)).squeeze(-1)
        sampled_probs = full_probs.gather(dim=2, index=sampled_tokens.unsqueeze(-1)).squeeze(-1)
        logits_tensor = sampled_logits
        probs_tensor = sampled_probs
    else:
        logits_tensor = full_logits
        probs_tensor = full_probs

    # For each sequence, find the first stop token (EOS or newline) to trim the generation.
    effective_lengths = []
    trimmed_sequences = []
    for i in range(batch_size):
        gen_tokens = generated_new_tokens[i]
        stop_index = num_steps  # default: no stop token found
        for t in range(num_steps):
            token = gen_tokens[t].item()
            if token == generation_config.eos_token_id or token == newline_token_id \
                    or token == newline_end_token_id:
                stop_index = t
                break
        effective_lengths.append(stop_index)
        trimmed_sequences.append(gen_tokens[:stop_index])

    # Trim logits/probabilities according to effective lengths.
    logits_list = [logits_tensor[i, :effective_lengths[i]] for i in range(batch_size)]
    probs_list = [probs_tensor[i, :effective_lengths[i]] for i in range(batch_size)]

    # Optionally pad outputs.
    if do_padding:
        padded_logits = pad_sequence(logits_list, batch_first=True, padding_value=0.0)
        padded_probs = pad_sequence(probs_list, batch_first=True, padding_value=0.0)
        max_length = padded_logits.size(1)
        batch_indices = torch.arange(max_length, device=cur_input_ids.device).unsqueeze(0).expand(batch_size, -1)
        effective_tensor = torch.tensor(effective_lengths, device=cur_input_ids.device).unsqueeze(1)
        attention_mask_out = batch_indices < effective_tensor

        return {
            "sequences": trimmed_sequences,
            "logits": padded_logits,
            "probabilities": padded_probs,
            "attention_mask": attention_mask_out,
        }
    else:
        return {
            "sequences": trimmed_sequences,
            "logits": logits_list,
            "probabilities": probs_list,
        }


def get_lora_model(model_name, lora_config):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


if __name__ == "__main__":
    # Load model and tokenizer.
    model_name = "/net/scratch2/listar2000/gfn-od/models/pretrained/Meta-Llama-3-8B-Instruct"
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        # target_modules=["q_proj", "k_proj", "v_proj"],
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model, tokenizer = get_lora_model(model_name, lora_config)

    prompt = "Generate one sentence about California:"
    # Create a custom generation configuration that stops on a newline.
    custom_gen_config = GenerationConfig(
        temperature=1.0,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        stop_strings=["\n"]
    )

    results = generate_sequences_with_logits(
        prompt,
        model,
        tokenizer,
        batch_size=4,
        max_new_tokens=50,
        sampled_only=True,
        generation_config=custom_gen_config
    )

    print("Generated sequences:")
    for seq in results["sequences"]:
        print(tokenizer.decode(seq, skip_special_tokens=False))
        print("--------------")

    print("\nLogits shape:", results["logits"].shape)
    print("Probabilities shape:", results["probabilities"].shape)
    print("Attention mask shape:", results["attention_mask"].shape)

    from replay_buffer import calculate_similarity_scores

    cache_dir = "/net/scratch2/listar2000/gfn-od/models/pretrained/sentence_transformer"
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=cache_dir)

    decoded_sequences = [tokenizer.decode(seq, skip_special_tokens=True) for seq in results["sequences"]]
    similarity_scores = calculate_similarity_scores(embedder, decoded_sequences)
    print("Similarity scores shape:", similarity_scores.shape)
    print("Similarity scores:", similarity_scores)

    # get the device of the embedder
    device = embedder.device
    print("Embedder device:", device)