import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from typing import Dict, List

class ReplayBuffer:
    def __init__(
        self,
        embedder: SentenceTransformer,
        method: str = "kmeans",
        threshold: int = 16 * 10,
        n_clusters: int = 2,
        dbscan_eps: float = 0.01,
        dbscan_min_samples: int = 1,
        retrain_every: int = 16 * 10
    ):
        """
        :param embedder: A SentenceTransformer to encode
        :param method: Which clustering method to use: 'kmeans', 'spectral', or 'dbscan'.
        :param threshold: Minimum number of embeddings before we train for a block.
        :param n_clusters: Number of clusters (for kmeans/spectral).
        :param dbscan_eps: eps parameter for DBSCAN (distance threshold).
        :param dbscan_min_samples: min_samples parameter for DBSCAN.
        :param retrain_every: Retrain the clustering every 'retrain_every' calls to update_replay_buffer.
        """
        self.embedder = embedder
        self.method = method.lower()
        self.threshold = threshold
        self.n_clusters = n_clusters
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.retrain_every = retrain_every

        # Store embeddings by block: { "block_name": List[np.ndarray], ... }
        self.embeddings: Dict[str, List[np.ndarray]] = {}

        # Store clusterers, labels, and cluster_sizes by block
        self.clusterers = {}
        self.labels = {}
        self.cluster_sizes = {}

        # Keep track of which blocks have been trained
        self.trained_blocks = set()

        # Internal counter to control retraining frequency
        self.global_step = 0

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts and return an array of shape (len(texts), embedding_dim).
        """
        return self.embedder.encode(texts)

    def update_replay_buffer(self, block: str, texts: List[str]):
        """
        High-level method to:
        1) Add new embeddings for the given 'block'.
        2) Possibly retrain clusters (if threshold is met and global_step % retrain_every == 0).
        """
        self.global_step += len(texts)

        # 1) Add new embeddings
        if block not in self.embeddings:
            self.embeddings[block] = []
        new_emb = self.get_embeddings(texts)
        new_emb = np.array(new_emb, dtype=np.float64)
        self.embeddings[block].extend(new_emb)

        # 2) Possibly retrain
        if (self.global_step % self.retrain_every == 0) and (len(self.embeddings[block]) >= self.threshold):
            self.train_clusters(block)

    def train_clusters(self, block: str):
        """
        Train/update clustering for a given block if threshold is met.
        - KMeans: direct on embeddings
        - SpectralClustering: build a similarity matrix from embeddings (cosine or RBF)
        - DBSCAN: directly on embeddings (Euclidean).
        """
        data = np.array(self.embeddings[block])
        if len(data) < self.threshold:
            raise ValueError(f"Cannot train clusters for block '{block}' with less than {self.threshold} embeddings.")

        if self.method == "kmeans":
            clusterer = KMeans(n_clusters=self.n_clusters, random_state=42)
            labels = clusterer.fit_predict(data)

        elif self.method == "spectral":
            clusterer = SpectralClustering(
                n_clusters=self.n_clusters,
                affinity="rbf",  # scikit-learn computes similarity internally
                random_state=42
            )
            labels = clusterer.fit_predict(data)

        elif self.method == "dbscan":
            clusterer = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
            labels = clusterer.fit_predict(data)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        # Save the clusterer & labels
        self.clusterers[block] = clusterer
        self.labels[block] = labels

        # Compute cluster sizes as {label -> count}
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_size_dict = dict(zip(unique_labels, counts))
        self.cluster_sizes[block] = cluster_size_dict

        self.trained_blocks.add(block)

    def get_probability(self, block: str, text: str) -> float:
        """
        Probability = (# points in assigned cluster) / (sum of points in all clusters)

        For KMeans: we use clusterer.predict().
        For SpectralClustering / DBSCAN: scikit-learn does not provide .predict().
          => We do a nearest-neighbor approach to figure out the cluster label:
             1) Find the nearest embedding in 'block'.
             2) Use that embedding's label.

        If block not trained, return None.
        """
        if block not in self.trained_blocks:
            # Not trained => cannot produce probability
            return None

        # Single-embed the text
        e = np.array(self.get_embeddings([text])[0]  , dtype=np.float64)
        labels = self.labels[block]
        data = np.array(self.embeddings[block])
        clusterer = self.clusterers[block]
        cluster_size_dict = self.cluster_sizes[block]

        # Sum of all clusters (including label=-1 if DBSCAN)
        total_points = sum(cluster_size_dict.values())

        # 1) If KMeans, we can directly predict
        if isinstance(clusterer, KMeans):
            label = clusterer.predict([e])[0]
        # 2) Otherwise, nearest-neighbor approach
        else:
            # Calculate Euclidean distances to training embeddings in 'block'
            diffs = data - e
            dists = np.sum(diffs**2, axis=1)  # shape (N,)
            nearest_idx = np.argmin(dists)
            label = labels[nearest_idx]

        # Probability = cluster_size[label] / total_points
        cluster_count = cluster_size_dict.get(label, 0)
        prob = cluster_count / float(total_points) if total_points > 0 else 0.0
        return prob


# ---------------------------------------------------------------------
# Example usage
if __name__ == "__main__":
    embedder = SentenceTransformer("bert-base-nli-mean-tokens")

    threshold_for_initial_training = 5
    buffer = ReplayBuffer(
        embedder,
        method="kmeans",   
        threshold=threshold_for_initial_training,         
        n_clusters=2,        
        retrain_every=  5
    )

    new_texts = [f"Example text {-1}-{j}" for j in range(5)]
    buffer.update_replay_buffer("first block", new_texts)

    for i in range(8):
        new_texts = [f"Example text {i}-{j}" for j in range(5)]
        buffer.update_replay_buffer("first block", new_texts)

        sample_text = "A sample text to test probability"
        prob = buffer.get_probability("first block", sample_text)
        print(f"Iteration {i}, Probability for sample text => {prob}")
        print("Cluster sizes:", buffer.cluster_sizes.get("first block", None))
