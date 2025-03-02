import torch
import torch.nn.functional as F

def compute_kl_penalty(gen_logits_list, ref_logits_list, device=None, debug=False):
    """
    Computes KL divergence between model logits and reference model logits, 
    ensuring that logits are truncated to the shortest sequence length instead of padding.

    Args:
        gen_logits_list (list of torch.Tensor): Logits from the fine-tuned model.
        ref_logits_list (list of torch.Tensor): Logits from the reference model.
        device (str, optional): Device to perform computations on (e.g., "cuda" or "cpu"). Defaults to auto-detection.
        debug (bool): If True, prints shapes and values for debugging.

    Returns:
        float: KL divergence loss (batch mean).
    """

    # Auto-detect device if not provided
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure we have valid logits
    if not gen_logits_list or not ref_logits_list:
        raise ValueError("Logits lists must not be empty.")

    # Move tensors to the correct device
    gen_logits_list = [logit.to(device) for logit in gen_logits_list]
    ref_logits_list = [logit.to(device) for logit in ref_logits_list]

    # **Ensure all sequences are truncated to the shortest length in the batch**
    min_seq_len = min(min(logit.shape[0] for logit in gen_logits_list), 
                      min(logit.shape[0] for logit in ref_logits_list))

    # Truncate all logits to min_seq_len
    gen_logits_list = [logit[:min_seq_len, :] for logit in gen_logits_list]
    ref_logits_list = [logit[:min_seq_len, :] for logit in ref_logits_list]

    # Convert lists to tensors
    model_logits = torch.stack(gen_logits_list).to(device)
    reference_logits = torch.stack(ref_logits_list).to(device)

    # Ensure vocab size matches
    if model_logits.shape[-1] != reference_logits.shape[-1]:
        raise ValueError(f"Vocabulary size mismatch: model_logits={model_logits.shape[-1]}, reference_logits={reference_logits.shape[-1]}")

    # Debugging logs
    if debug:
        print(f"Model logits shape: {model_logits.shape}")
        print(f"Reference logits shape: {reference_logits.shape}")
        print(f"Model logits:\n{model_logits}")
        print(f"Reference logits:\n{reference_logits}")

    # Convert logits to log-probabilities for numerical stability
    model_log_probs = F.log_softmax(model_logits, dim=-1)
    reference_probs = F.softmax(reference_logits, dim=-1)

    # Compute KL divergence
    kl_penalty = F.kl_div(model_log_probs, reference_probs, reduction="batchmean")

    return kl_penalty.item()


### **🔹 Deterministic Test Case 🔹 ###
if __name__ == "__main__":
    # Define manual logits for reproducibility (Different sequence lengths)
    gen_logits = [
        torch.tensor([[2.0, 1.0, 0.1], [3.0, 2.0, 0.5], [0.5, 0.2, 0.3]]),  # seq_len=3, vocab_size=3
        torch.tensor([[1.5, 2.5, 0.3], [2.2, 1.8, 0.7]])  # seq_len=2, vocab_size=3
    ]

    ref_logits = [
        torch.tensor([[1.8, 1.2, 0.3], [2.9, 2.1, 0.4], [0.4, 0.1, 0.2]]),  # seq_len=3, vocab_size=3
        torch.tensor([[1.4, 2.4, 0.4], [2.0, 1.9, 0.6], [1.0, 1.0, 1.0]])  # seq_len=3, vocab_size=3
    ]

    # Compute KL penalty
    kl_loss = compute_kl_penalty(gen_logits, ref_logits, debug=True)
    print(f"KL Penalty: {kl_loss:.6f}")
