from sentence_transformers import SentenceTransformer
from typing import List
import torch

# some utility functions
def calculate_similarity_scores(model: SentenceTransformer, texts: List[str]) -> torch.Tensor:
    embeddings = model.encode(texts)
    return model.similarity(embeddings, embeddings).to(model.device)

def get_effective_probabilities(similarity_scores: torch.Tensor, raw_log_probs: torch.Tensor) -> torch.Tensor:
    # step 1: scale the raw_probs so they sum to 1
    max_log_prob = raw_log_probs.max()

    log_Z = max_log_prob + torch.log(torch.exp(raw_log_probs - max_log_prob).sum())
    log_Z_detached = log_Z.detach()

    scaled_log_probs = raw_log_probs - log_Z_detached
    scaled_probs = torch.exp(scaled_log_probs)
    # step 2: apply the similarity scores
    similarity_scores = similarity_scores.to(scaled_probs.dtype)
    return torch.matmul(similarity_scores, scaled_probs)

if __name__ == "__main__":
    fake_similarity_scores = torch.tensor([[1, 0.5, 0.2], [0.5, 1, 0.2], [0.2, 0.2, 1]])
    fake_raw_probs = torch.tensor([0.9, 0.81, 0.02])
    effective_probs = get_effective_probabilities(fake_similarity_scores, fake_raw_probs)
    print(effective_probs)