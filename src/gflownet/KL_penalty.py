import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def compute_kl_penalty(gen_logits_list, ref_logits_list, device="cpu"):
    """
    Computes KL divergence between model logits and reference model logits.

    Args:
        gen_logits_list (list of torch.Tensor): Logits from the fine-tuned model.
        ref_logits_list (list of torch.Tensor): Logits from the reference model.
        device (str): Device to perform computations on.

    Returns:
        float: KL divergence loss (batch mean).
    """

    # Ensure we have valid logits
    if not gen_logits_list or not ref_logits_list:
        raise ValueError("Logits lists must not be empty.")

    # Move tensors to device
    gen_logits_list = [logit.to(device) for logit in gen_logits_list]
    ref_logits_list = [logit.to(device) for logit in ref_logits_list]

    # Pad sequences before stacking
    model_logits = pad_sequence(gen_logits_list, batch_first=True, padding_value=0).to(device)
    reference_logits = pad_sequence(ref_logits_list, batch_first=True, padding_value=0).to(device)

    print(model_logits.shape)
    print(reference_logits.shape)
    print(model_logits)
    print(reference_logits)

    # Ensure both tensors have the same sequence length
    min_seq_len = min(model_logits.shape[1], reference_logits.shape[1])


    # Truncate both tensors to match lengths
    model_logits = model_logits[:, :min_seq_len, :]
    reference_logits = reference_logits[:, :min_seq_len, :]

    # Convert logits to log-probabilities for numerical stability
    model_log_probs = F.log_softmax(model_logits, dim=-1)
    reference_probs = F.softmax(reference_logits, dim=-1)

    # Compute KL divergence
    kl_penalty = F.kl_div(model_log_probs, reference_probs, reduction="batchmean")

    return kl_penalty.item()

if __name__ == "__main__":
    # Define manual logits for reproducibility
    gen_logits = [
        torch.tensor([[2.0, 1.0, 0.1], [3.0, 2.0, 0.5]]),  # Example 1 (seq_len=2, vocab_size=3)
        torch.tensor([[1.5, 2.5, 0.3], [2.2, 1.8, 0.7], [0.9, 1.1, 1.3]])  # Example 2 (seq_len=3, vocab_size=3)
    ]

    ref_logits = [
        torch.tensor([[1.8, 1.2, 0.3], [2.9, 2.1, 0.4]]),  # Reference Example 1
        torch.tensor([[1.4, 2.4, 0.4], [2.0, 1.9, 0.6], [1.0, 1.0, 1.0]])  # Reference Example 2
    ]

    # Compute KL penalty
    kl_loss = compute_kl_penalty(gen_logits, ref_logits, device="cpu")
    print(f"KL Penalty: {kl_loss:.6f}")