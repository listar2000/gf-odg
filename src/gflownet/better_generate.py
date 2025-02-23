import torch
import torch.nn.functional as F
from transformers import GenerationConfig
from peft import LoraConfig
from model import get_lora_model


def generate_sequences_with_logits(
    prompt,
    model,
    tokenizer,
    batch_size=4,
    max_new_tokens=20,
    generation_config=None
):
    # --- Parse generation configuration ---
    temperature = generation_config.temperature if generation_config and hasattr(generation_config, 'temperature') else 1.0
    top_p = generation_config.top_p if generation_config and hasattr(generation_config, 'top_p') else 1.0
    do_sample = generation_config.do_sample if generation_config and hasattr(generation_config, 'do_sample') else False
    eos_token_id = generation_config.eos_token_id if generation_config and hasattr(generation_config, 'eos_token_id') else None
    stop_strings = generation_config.stop_strings if generation_config and hasattr(generation_config, 'stop_strings') else []

    device = next(model.parameters()).device

    # --- Tokenize the prompt and replicate for the batch ---
    encoded = tokenizer(prompt, return_tensors="pt")
    prompt_ids = encoded.input_ids.to(device)  # shape: (1, seq_len)
    prompt_ids = prompt_ids.repeat(batch_size, 1)  # shape: (batch_size, seq_len)

    # We'll store the generated sequences as a list of tensors.
    generated_sequences = [prompt_ids[i].clone() for i in range(batch_size)]
    # For recording logits and probabilities per generation step (per sample).
    logits_list = [[] for _ in range(batch_size)]
    probs_list = [[] for _ in range(batch_size)]

    # Active mask: True means the sequence is still generating.
    active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    # --- Initial forward pass with the full prompt ---
    with torch.autocast(device.type, dtype=torch.bfloat16):
        outputs = model(input_ids=prompt_ids, use_cache=True)
    # Get initial logits from the prompt's last token (unused for generation, but we still record nothing here)
    batched_cache = outputs.past_key_values  # native cached structure for the full batch

    # Compute prompt length for later trimming.
    prompt_length = prompt_ids.shape[1]

    # --- Generation loop ---
    for step in range(max_new_tokens):
        # For each sequence in the batch, decide the next input token:
        # If active, use the last token of the generated sequence;
        # if finished, feed eos_token_id (so that the cache remains consistent).
        next_input_tokens = []
        for i in range(batch_size):
            if active_mask[i]:
                next_input_tokens.append(generated_sequences[i][-1].unsqueeze(0))
            else:
                # If finished, force eos_token (or any token, since its output will be ignored)
                next_input_tokens.append(torch.tensor([eos_token_id], device=device))
        # Stack into a tensor of shape (batch_size, 1)
        next_input = torch.stack(next_input_tokens, dim=0)

        # Run forward pass using only the last token and the batched cache.
        with torch.autocast(device.type, dtype=torch.bfloat16):
            outputs = model(input_ids=next_input, past_key_values=batched_cache, use_cache=True)
        # outputs.logits: shape (batch_size, 1, vocab_size)
        logits = outputs.logits[:, -1, :]  # shape: (batch_size, vocab_size)
        batched_cache = outputs.past_key_values  # update batched cache for next step

        # Process logits per sample:
        for i in range(batch_size):
            # For finished sequences, force token to be eos.
            if not active_mask[i]:
                token_id = eos_token_id
                token_logit = logits[i]
                token_prob = F.softmax(token_logit, dim=-1)[eos_token_id].unsqueeze(0)
            else:
                scaled_logits = logits[i] / temperature
                if do_sample:
                    probs = F.softmax(scaled_logits, dim=-1)
                    # Top-p filtering
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    filtered_logits = scaled_logits.clone()
                    remove_mask = torch.zeros_like(filtered_logits, dtype=torch.bool)
                    remove_mask.scatter_(0, sorted_indices, sorted_indices_to_remove)
                    filtered_logits.masked_fill_(remove_mask, float('-inf'))
                    filtered_probs = F.softmax(filtered_logits, dim=-1)
                    token_id = torch.multinomial(filtered_probs, num_samples=1).item()
                    token_prob = filtered_probs[token_id].unsqueeze(0)
                    token_logit = logits[i]
                else:
                    token_id = torch.argmax(scaled_logits, dim=-1).item()
                    token_prob = F.softmax(scaled_logits, dim=-1)[token_id].unsqueeze(0)
                    token_logit = logits[i]
            
            # Append the new token to the generated sequence.
            new_token_tensor = torch.tensor([token_id], device=device)
            generated_sequences[i] = torch.cat([generated_sequences[i], new_token_tensor])

            # Record logits and probability for this step.
            logits_list[i].append(token_logit.unsqueeze(0))  # shape (1, vocab_size)
            probs_list[i].append(token_prob.unsqueeze(0))      # shape (1, 1)

            # Check for termination if active.
            if active_mask[i]:
                if eos_token_id is not None and token_id == eos_token_id:
                    active_mask[i] = False
                elif stop_strings:
                    # Decode current sequence and check for any stop string.
                    current_text = tokenizer.decode(generated_sequences[i], skip_special_tokens=True)
                    if any(stop in current_text for stop in stop_strings):
                        active_mask[i] = False

        # If all sequences are finished, break early.
        if not active_mask.any():
            break

    # --- Postprocess: trim sequences after the first eos token (if present) ---
    trimmed_sequences = []
    for seq in generated_sequences:
        seq_list = seq.tolist()
        if eos_token_id is not None and eos_token_id in seq_list:
            first_eos = seq_list.index(eos_token_id)
            seq = seq[:first_eos+1]
        trimmed_sequences.append(seq)

    # Concatenate per-step logits and probabilities per sample.
    logits_list = [torch.cat(steps, dim=0) if steps else None for steps in logits_list]
    probs_list = [torch.cat(steps, dim=0) if steps else None for steps in probs_list]

    # Optionally, cast outputs back to float32.
    logits_list = [t.float() if t is not None else None for t in logits_list]
    probs_list = [t.float() if t is not None else None for t in probs_list]

    return {
        "sequences": trimmed_sequences,
        "logits": logits_list,
        "probabilities": probs_list,
    }


def benchmark_generation(
    prompt,
    model,
    tokenizer,
    batch_size=4,
    max_new_tokens=20,
    generation_config=None,
    num_runs=10
):  
    import time
    from tqdm import tqdm
    # Warm-up runs to account for any initial overhead (e.g. JIT/autocast warm-up)
    for _ in range(2):
        generate_sequences_with_logits(
            prompt, model, tokenizer, batch_size=batch_size, max_new_tokens=max_new_tokens, generation_config=generation_config
        )
    torch.cuda.synchronize()  # Ensure GPU is ready

    total_tokens = 0
    total_time = 0.0

    # Compute the prompt length (we assume all outputs share the same prompt)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_length = prompt_ids.shape[1]

    for run in tqdm(range(num_runs)):
        start_time = time.time()
        result = generate_sequences_with_logits(
            prompt, model, tokenizer, batch_size=batch_size, max_new_tokens=max_new_tokens, generation_config=generation_config
        )
        torch.cuda.synchronize()  # Wait for GPU computations to finish
        elapsed = time.time() - start_time

        # Calculate the number of newly generated tokens per sample.
        for seq in result["sequences"]:
            # Subtract prompt length to get new tokens
            new_tokens = len(seq) - prompt_length
            total_tokens += new_tokens

        total_time += elapsed
        print(f"Run {run+1}/{num_runs}: {elapsed:.3f} sec, new tokens per run: {batch_size * (max_new_tokens)} (approx.)")

        # sample outputs
        # for seq in result["sequences"]:
        #     print("Sample output: -------------------------------------------------")
        #     print(tokenizer.decode(seq, skip_special_tokens=True))
        #     print("-------------------------------------------------")
        #     break

    tokens_per_sec = total_tokens / total_time
    print(f"\nProcessed {total_tokens} tokens in {total_time:.3f} sec, throughput: {tokens_per_sec:.2f} tokens/sec")


# Example usage:
if __name__ == "__main__":
    # Assume you have a function `get_lora_model` that returns your model and tokenizer.
    # Also assume you have defined your `GenerationConfig` accordingly.
    model_name = "/net/scratch2/listar2000/gfn-od/models/pretrained/Meta-Llama-3-8B-Instruct"
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj"],
        # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model, tokenizer = get_lora_model(model_name, lora_config)
    prompt = "Once upon a time, "

    generation_config = GenerationConfig(
        temperature=1.0,
        top_p=1.0,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        stop_strings=["\n"]
    )

    benchmark_generation(prompt, model, tokenizer, batch_size=32, max_new_tokens=30, generation_config=generation_config, num_runs=10)
