from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer
from state import Concept
from interceptor import RawTextProcessor


def get_lora_model(model_name_or_path, lora_r=8, lora_alpha=32, lora_dropout=0.05):
    """
    Load a model with LoRA configuration and return model, tokenizer, text_processor, and sentence_transformer.
    
    Args:
        model_name_or_path: Path to the model or model name
        lora_r: LoRA r dimension
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        
    Returns:
        model: The loaded model with LoRA configuration
        tokenizer: The tokenizer for the model
        text_processor: A text processor for extracting concepts
        sentence_transformer: A sentence transformer for embedding text
    """
    # Set up paths
    MODEL_DIR = "/net/scratch2/listar2000/gfn-od/models/"
    cache_dir = MODEL_DIR + "pretrained/sentence_transformer"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load sentence transformer
    sentence_transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=cache_dir)
    
    return model, tokenizer, sentence_transformer


if __name__ == "__main__":
    # Load model and tokenizer.
    model_name = "/net/scratch2/listar2000/gfn-od/models/pretrained/Meta-Llama-3-8B-Instruct"
    model, tokenizer, text_processor, sentence_transformer = get_lora_model(model_name)

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

    decoded_sequences = [tokenizer.decode(seq, skip_special_tokens=True) for seq in results["sequences"]]
    similarity_scores = calculate_similarity_scores(sentence_transformer, decoded_sequences)
    print("Similarity scores shape:", similarity_scores.shape)
    print("Similarity scores:", similarity_scores)

    # get the device of the embedder
    device = sentence_transformer.device
    print("Embedder device:", device)