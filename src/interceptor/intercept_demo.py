import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.interceptor.trigger_interceptor import DecodeInterceptor


def load_model_and_tokenizer(config):
    """Load the Gemma model and tokenizer based on config."""
    model_config = config.models.gemma_2b

    print(f"Loading model from {model_config.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_path,
        **model_config.model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_path)

    return model, tokenizer


def setup_interceptor(tokenizer):
    """Set up the decode interceptor with fruit-related trigger words."""
    interceptor_config = {
        "trigger_interceptor": {
            "trigger_words": [
                # Common fruits with various forms
                # Basic forms
                "apple", "apples",
                "banana", "bananas",
                "orange", "oranges",
                "grape", "grapes",
                "strawberry", "strawberries",
                "berry", "berries",
                "kiwi", "kiwis",
                "pear", "pears",
                "peach", "peaches",
                "pineapple", "pineapples",
                "mango", "mangoes",
                "watermelon", "watermelons",
            ],
            "enabled": True,
            "stop_on_trigger": True,
            "case_sensitive": True,
            "case_variants": ["original", "capitalized", "upper"]
        }
    }
    config = OmegaConf.create(interceptor_config)
    return DecodeInterceptor(tokenizer, config)


def generate_with_interceptor(model, tokenizer, interceptor, prompt, max_length=100):
    """Generate text with the interceptor active."""
    # Tokenize input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)

    # Get generation config from model
    generation_config = model.generation_config
    generation_config.max_length = max_length

    # Track generated tokens and their triggers
    generated_tokens = []
    triggers_found = []

    # Generate tokens one by one
    cur_length = input_ids.shape[1]
    while cur_length < max_length:
        outputs = model.generate(
            input_ids,
            max_new_tokens=1,
            pad_token_id=tokenizer.pad_token_id,
            generation_config=generation_config
        )

        # Get the new token
        new_token = outputs[:, -1:]

        # Print token and its decoded form for debugging
        print(new_token[0], f"'{tokenizer.decode(new_token[0].tolist())}'")

        # Check for triggers
        results = interceptor.intercept(new_token)
        for result in results:
            if result.metadata:
                trigger = result.metadata["trigger_word"]
                triggers_found.append((
                    trigger,
                    tokenizer.decode(new_token[0].tolist())
                ))
                if result.should_stop:
                    print(
                        f"\nStopping generation due to trigger word: {trigger}")
                    return tokenizer.decode(generated_tokens), triggers_found

        # Add the new token and continue
        generated_tokens.extend(new_token[0].tolist())
        input_ids = outputs
        cur_length += 1

    return tokenizer.decode(generated_tokens), triggers_found


def main():
    # Load configurations
    config = OmegaConf.load("configs/main.yaml")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Setup interceptor
    interceptor = setup_interceptor(tokenizer)

    # Test prompts
    prompts = [
        "Do you like apple?",
        "Where do people get banana from?",
        "Tell me about strawberry:",
    ]

    # Generate with each prompt
    for i, prompt in enumerate(prompts, 1):
        print(f"\n=== Test {i}: '{prompt}' ===")
        generated_text, triggers = generate_with_interceptor(
            model, tokenizer, interceptor, prompt)

        print("\nGenerated Text:")
        print(generated_text)

        if triggers:
            print("\nTriggers Found:")
            for trigger, context in triggers:
                print(f"- '{trigger}' (in context: '{context}')")
        else:
            print("\nNo triggers found.")

        # reset interceptor
        interceptor.reset()


if __name__ == "__main__":
    main()
