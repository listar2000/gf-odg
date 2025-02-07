import pytest
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.interceptor.trigger_interceptor import TriggerInterceptor


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def basic_config():
    config_dict = {
        "trigger_interceptor": {
            "trigger_words": ["hello", "Bad Word"],
            "enabled": True,
            "stop_on_trigger": True,
            "case_sensitive": False,
            "case_variants": ["original", "capitalized"]
        }
    }
    return OmegaConf.create(config_dict)


def test_trigger_variants(tokenizer, basic_config):
    """Test that different case variants of trigger words are correctly detected."""
    interceptor = TriggerInterceptor(tokenizer, basic_config)

    # Get tokenized variants for verification
    variants = {}
    for trigger in basic_config.trigger_interceptor.trigger_words:
        variants[trigger] = []
        trigger_variants = interceptor._get_case_variants(
            trigger, basic_config.trigger_interceptor.case_variants)
        for variant in trigger_variants:
            tokens = tokenizer.encode(variant, add_special_tokens=False)
            variants[trigger].append((variant, tokens))

    # Verify each variant
    for trigger, variant_list in variants.items():
        print(f"\nTesting variants for '{trigger}':")
        for variant, tokens in variant_list:
            print(f"  Testing variant '{variant}' with tokens {tokens}")
            results = []
            
            # Feed tokens one by one
            for token in tokens:
                # Create a batch with a single token
                curr_tokens = torch.tensor([[token]])  # Shape: (1, 1)
                step_results = interceptor.intercept(curr_tokens)
                results.extend(step_results)
            
            # The last result should contain the trigger
            assert results[-1].metadata is not None, f"Failed to detect variant '{variant}'"
            detected_variant = results[-1].metadata["trigger_word"]
            print(f"  Detected: {detected_variant}")
            assert detected_variant == variant, f"Expected '{variant}', got '{detected_variant}'"


def test_batch_processing(tokenizer, basic_config):
    """Test that the interceptor correctly handles batch processing."""
    interceptor = TriggerInterceptor(tokenizer, basic_config)

    # Create a batch of 3 sequences
    eos_token = tokenizer.eos_token_id

    # Get token ids for test words
    hello_ids = tokenizer.encode("Hello", add_special_tokens=False)
    bad_word_ids = tokenizer.encode("Bad Word", add_special_tokens=False)

    # Define generation steps
    generation_steps = [
        # Step 1: normal token, start of "Bad Word", normal token
        [100, bad_word_ids[0], 50],
        # Step 2: "Hello", complete "Bad Word", normal token
        [hello_ids[0], bad_word_ids[1], 101],
        # Step 3: EOS, normal token, EOS
        [eos_token, 102, eos_token]
    ]

    print("\nTesting batch processing:")
    for step_idx, step_tokens in enumerate(generation_steps, start=1):
        # Convert the list of tokens to a tensor of shape (batch_size, 1)
        new_tokens_tensor = torch.tensor([[token] for token in step_tokens])
        results = interceptor.intercept(new_tokens_tensor)

        print(f"\nStep {step_idx}, new tokens: {step_tokens}")
        for i, res in enumerate(results):
            print(
                f"  Sample {i}: should_stop={res.should_stop}, metadata={res.metadata}")

            # Verify specific cases
            if step_idx == 2:
                if i == 0:  # First sequence should detect "Hello"
                    assert res.metadata is not None, "Failed to detect 'Hello'"
                    assert res.metadata["trigger_word"] == "Hello"
                elif i == 1:  # Second sequence should detect "Bad Word"
                    assert res.metadata is not None, "Failed to detect 'Bad Word'"
                    assert res.metadata["trigger_word"] == "Bad Word"
                else:  # Third sequence should have no trigger
                    assert res.metadata is None, "Unexpected trigger detected"


def test_case_sensitivity(tokenizer):
    """Test that case sensitivity setting is respected."""
    # Test with case sensitivity enabled
    sensitive_config = OmegaConf.create({
        "trigger_interceptor": {
            "trigger_words": ["Hello"],
            "enabled": True,
            "stop_on_trigger": True,
            "case_sensitive": True
        }
    })

    sensitive_interceptor = TriggerInterceptor(tokenizer, sensitive_config)

    # Test with different cases
    test_words = ["Hello", "hello", "HELLO"]
    for word in test_words:
        tokens = torch.tensor(
            [[t] for t in tokenizer.encode(word, add_special_tokens=False)])
        results = []
        for i in range(tokens.shape[0]):
            curr_tokens = tokens[i:i+1]
            step_results = sensitive_interceptor.intercept(curr_tokens)
            results.extend(step_results)

        # Only "Hello" should be detected
        should_detect = word == "Hello"
        last_result = results[-1]

        print(f"\nTesting case sensitive detection of '{word}':")
        print(f"  Should detect: {should_detect}")
        print(f"  Result: {last_result.metadata}")

        if should_detect:
            assert last_result.metadata is not None, f"Failed to detect '{word}'"
            assert last_result.metadata["trigger_word"] == "Hello"
        else:
            assert last_result.metadata is None, f"Incorrectly detected '{word}'"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
