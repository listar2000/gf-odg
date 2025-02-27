from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer


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