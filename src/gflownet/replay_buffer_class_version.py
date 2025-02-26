import torch
from sklearn.cluster import SpectralClustering, DBSCAN

def get_effective_probabilities_spectral(similarity_scores: torch.Tensor, n_clusters=2) -> torch.Tensor:
    similarity_matrix = similarity_scores.numpy()
    
    # Spectral Clustering
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='discretize')
    labels = clustering.fit_predict(similarity_matrix)
    print(labels)

    # Count the number of items in each cluster and calculate probabilities
    cluster_sizes = torch.bincount(torch.tensor(labels, dtype=torch.int64), minlength=n_clusters)
    probabilities = torch.tensor([cluster_sizes[label] / len(labels) for label in labels], dtype=torch.float32)
    
    return probabilities

def get_effective_probabilities_threshold(similarity_scores: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    mask = similarity_scores > threshold
    n = similarity_scores.size(0)
    probabilities = torch.zeros(n, dtype=torch.float32)
    
    # Calculate probabilities for each "connected component"
    visited = torch.zeros(n, dtype=torch.bool)
    for i in range(n):
        if not visited[i]:
            connected_indices = mask[i].nonzero(as_tuple=False).squeeze(1)
            cluster_size = len(connected_indices)
            probabilities[connected_indices] = cluster_size / n
            visited[connected_indices] = True
    
    return probabilities

def get_effective_probabilities_dbscan(similarity_scores: torch.Tensor) -> torch.Tensor:
    distance_matrix = 1 - similarity_scores.numpy()  # DBSCAN works with distance

    # DBSCAN clustering
    clustering = DBSCAN(eps=0.3, min_samples=1, metric="precomputed")
    labels = clustering.fit_predict(distance_matrix)
    print(labels)

    # Count the number of items in each cluster and calculate probabilities
    unique_labels, counts = torch.unique(torch.tensor(labels), return_counts=True)
    probabilities = torch.zeros(len(labels), dtype=torch.float32)
    for label, count in zip(unique_labels, counts):
        if label != -1:  # Ignore noise points
            indices = (torch.tensor(labels) == label).nonzero(as_tuple=False).squeeze(1)
            probabilities[indices] = count / len(labels)

    return probabilities

if __name__ == "__main__":
    fake_similarity_scores = torch.tensor([[1, 0.5, 0.2], [0.5, 1, 0.2], [0.2, 0.2, 1]])

    spectral_probs = get_effective_probabilities_spectral(fake_similarity_scores)
    print("Spectral:", spectral_probs)
    threshold_probs = get_effective_probabilities_threshold(fake_similarity_scores)

    print("Threshold:", threshold_probs)

    dbscan_probs = get_effective_probabilities_dbscan(fake_similarity_scores)
    print("DBSCAN:", dbscan_probs)
