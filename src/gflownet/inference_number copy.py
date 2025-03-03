import torch
import csv
import argparse
from transformers import GenerationConfig, AutoTokenizer
from peft import PeftModel
from model import get_lora_model
from better_generate import generate_sequences_with_logits
from interceptor import RawTextProcessor
from state import Concept, ConceptBlock
from dataclasses import dataclass
from typing import Optional

# Define TextGenerationConfig to match training script
@dataclass
class TextGenerationConfig:
    """Configuration for text generation."""
    prompt: str
    max_new_tokens: int = 40
    batch_size: int = 32
    generation_config: Optional[GenerationConfig] = None

def extract_numbers_from_concepts(sequences, tokenizer, text_processor):
    """
    Extract numbers from ConceptBlocks in generated sequences.
    """
    extracted_numbers = []
    for sequence in sequences:
        decoded_texts = [tokenizer.decode(token, skip_special_tokens=True) for token in sequence]
        trajectory, _ = text_processor.process_text_to_trajectory(decoded_texts)
        print(decoded_texts)

        # Collect only the numbers (options) from ConceptBlocks
        numbers = [block.option for block in trajectory if isinstance(block, ConceptBlock)]
        extracted_numbers.append(",".join(numbers))  # Convert list to comma-separated string
        print (extracted_numbers)
    return extracted_numbers

def run_inference(model, tokenizer, gen_config, text_processor, N, output_csv, model_type):
    """
    Run inference N times and save extracted numbers to CSV.
    """
    results = []
    for i in range(N // gen_config.batch_size):  # Loop through in batches
        print(f"\nRunning batch {i + 1}/{N // gen_config.batch_size} for {model_type} model...")

        # Generate sequences
        generations = generate_sequences_with_logits(
            prompt=gen_config.prompt,
            model=model,
            tokenizer=tokenizer,
            batch_size=gen_config.batch_size,
            max_new_tokens=gen_config.max_new_tokens,
            generation_config=gen_config.generation_config
        )

        extracted_numbers = extract_numbers_from_concepts(generations["sequences"], tokenizer, text_processor)

        for j in range(gen_config.batch_size):
            results.append([i * gen_config.batch_size + j + 1, model_type, extracted_numbers[j]])
            print(f"\nRun {i * gen_config.batch_size + j + 1} ({model_type}): {extracted_numbers[j]}")

    # Save results to CSV
    with open(output_csv, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Run", "Model Type", "Extracted Numbers"])
        writer.writerows(results)

    print(f"\nInference completed for {model_type}. Results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on both fine-tuned and base models")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the fine-tuned LoRA adapter")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--prompt", type=str, default="Generate 6 random numbers from 1 to 6, separated by commas:", help="Input prompt for generation")
    parser.add_argument("--N", type=int, default=50, help="Total number of inferences to run")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--output_csv_base", type=str, default="inference_base.csv", help="CSV file for base model results")
    parser.add_argument("--output_csv_adapter", type=str, default="inference_finetuned.csv", help="CSV file for fine-tuned model results")

    args = parser.parse_args()

    # Load base model
    #print("\nLoading base model...")
    #base_model, tokenizer, _, _ = get_lora_model(model_name_or_path=args.base_model_path)

    # Load fine-tuned (adapter) model correctly
    print("\nLoading fine-tuned LoRA adapter...")
    adapter_model, tokenizer, _, _ = get_lora_model(model_name_or_path=args.base_model_path)
    adapter_model = PeftModel.from_pretrained(adapter_model, args.adapter_path)  # ✅ Fixed


    # Define concepts for number extraction
    ListOfNumbers = ['1', '2', '3', '4', '5', '6']
    N_Concepts = 6
    ConceptNames = [Concept(f"Number{i+1}", ListOfNumbers, case_variants=["capitalized", "lower", "plural"]) for i in range(N_Concepts)]
    text_processor = RawTextProcessor(ConceptNames, max_window_size=N_Concepts, only_concepts=True)

    # Configure text generation settings
    gen_config = TextGenerationConfig(
        prompt=args.prompt,
        batch_size=args.batch_size,
        max_new_tokens=20,
        generation_config=GenerationConfig(
            temperature=1.0,
            top_p=0.95,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            stop_strings=["\n", ".\n\n", ".\n"]
        )
    )

    # Run inference for both models
    #run_inference(base_model, tokenizer, gen_config, text_processor, args.N, args.output_csv_base, "Base Model")
    run_inference(adapter_model, tokenizer, gen_config, text_processor, args.N, args.output_csv_adapter, "Fine-Tuned Model")

