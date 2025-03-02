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
    MODEL_DIR = "/net/scratch/jiaweizhang"
    cache_dir = MODEL_DIR 
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    reference_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    reference_model.eval()

    
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
    
    return model, tokenizer, sentence_transformer, reference_model

