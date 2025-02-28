import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class DiversityReplayBuffer:
    """
    A replay buffer specifically designed for diversity training.
    Maintains separate buffers for different concept options and clusters open-ended responses.
    """
    def __init__(
        self,
        embedder: SentenceTransformer,
        n_clusters: int = 5,
        max_n_clusters: int = 10,
        fixed_n_clusters: bool = True,
        buffer_size: int = 1000,
        update_clusters_every: int = 100,
        min_samples_for_clustering: int = 50,
        random_state: int = 42
    ):
        """
        Initialize the diversity replay buffer.
        
        Args:
            embedder: SentenceTransformer for embedding text
            n_clusters: Number of clusters for open-ended responses
            max_n_clusters: Maximum number of clusters
            fixed_n_clusters: If True, n_clusters is fixed and cannot be changed
            buffer_size: Maximum size of each buffer
            update_clusters_every: Update clusters after this many additions
            min_samples_for_clustering: Minimum number of samples required before clustering
        """
        self.embedder = embedder
        self.n_clusters = n_clusters
        self.max_n_clusters = max_n_clusters
        self.fixed_n_clusters = fixed_n_clusters
        self.buffer_size = buffer_size
        self.update_clusters_every = update_clusters_every
        self.min_samples_for_clustering = min_samples_for_clustering
        self.random_state = random_state
        
        # Buffers for different concept options
        # {concept_option: {
        #     'texts': List[str],
        #     'embeddings': torch.Tensor,
        #     'clusters': Optional[KMeans],
        #     'cluster_centers': Optional[torch.Tensor],
        #     'update_counter': int
        # }}
        self.option_buffers = {}
        
        # Device for computations
        self.device = next(embedder.parameters()).device
        
    def add_samples(
        self,
        concept_option: str,
        texts: List[str],
        prefill: bool = False
    ) -> np.ndarray:
        """
        Add samples to the buffer for a specific concept option.
        
        Args:
            concept_option: The concept option (e.g., 'cat', 'dog')
            texts: List of open-ended response texts
            probs: Optional list of probability tensors for each text
        """
        if not texts:
            raise ValueError("texts must be non-empty")
            
        # Initialize buffer for this option if it doesn't exist
        if concept_option not in self.option_buffers:
            self.option_buffers[concept_option] = {
                'texts': [],
                'embeddings': None,
                'clusters': None,
                'cluster_centers': None,
                'update_counter': 0
            }
            
        buffer = self.option_buffers[concept_option]
        
        embeddings = self.embedder.encode(texts)

        # Add to buffer
        buffer['texts'].extend(texts)
        
        # Update embeddings tensor
        if buffer['embeddings'] is None:
            buffer['embeddings'] = embeddings
        else:
            # buffer['embeddings'] = torch.cat([buffer['embeddings'], embeddings], dim=0)
            buffer['embeddings'] = np.concatenate([buffer['embeddings'], embeddings], axis=0)

        # Update counter
        buffer['update_counter'] += len(texts)
        
        flag_1 = (buffer['update_counter'] >= self.update_clusters_every and 
            len(buffer['texts']) >= self.min_samples_for_clustering)
        flag_2 = (buffer['clusters'] is None)
        # Update clusters if needed
        if (flag_1 or flag_2) and not prefill:
            # Trim buffer if it exceeds maximum size
            if len(buffer['texts']) > self.buffer_size:
                # buffer['texts'] = buffer['texts'][-self.buffer_size:]
                # buffer['embeddings'] = buffer['embeddings'][-self.buffer_size:]
                self.balanced_evict(buffer)

            self._update_clusters(concept_option)
            buffer['update_counter'] = 0

        if buffer['clusters'] is not None:
            assigned = self.assign_cluster(concept_option, texts)
            return assigned
        else:
            return None

    def balanced_evict(self, buffer: dict):
        total_samples = len(buffer['texts'])
        if total_samples <= self.buffer_size:
            return
        
        # Predict clusters for all samples
        labels = buffer['clusters'].predict(buffer['embeddings'])
        n_clusters = buffer['clusters'].n_clusters
        
        # Determine quota per cluster
        quota = self.buffer_size // n_clusters
        new_texts, buffer_texts = [], []
        new_embeddings, buffer_embeddings = [], []
        # Track counts per cluster during retention
        cluster_counts = {i: 0 for i in range(n_clusters)}
        
        global_counter = 0

        zipped = list(zip(buffer['texts'], buffer['embeddings'], labels))
        # Loop through samples in FIFO order (or keep an index ordering)
        for text, emb, label in reversed(zipped):
            if cluster_counts[label] < quota:
                new_texts.append(text)
                new_embeddings.append(emb)
                cluster_counts[label] += 1
                global_counter += 1
            else:
                buffer_texts.append(text)
                buffer_embeddings.append(emb)

        remain_count = self.buffer_size - global_counter

        # Update buffer with balanced samples
        buffer['texts'] = new_texts + buffer_texts[:remain_count]
        buffer['embeddings'] = np.array(new_embeddings + buffer_embeddings[:remain_count])
        logger.info(f"Balanced evicted {total_samples - self.buffer_size} samples, \
            each cluster has at least {quota} samples, remaining {remain_count} samples")
            
    def _update_clusters(self, concept_option: str) -> None:
        """
        Update clusters for a specific concept option.
        
        Args:
            concept_option: The concept option to update clusters for
        """
        buffer = self.option_buffers[concept_option]
        
        # Skip if not enough samples
        if len(buffer['texts']) < self.min_samples_for_clustering:
            return
            
        # Convert embeddings to numpy for sklearn
        embeddings_np = buffer['embeddings']

        best_k = None
        best_score = -1
        best_model = None

        if not self.fixed_n_clusters:
            # Try a range of K values (for instance, from 2 to some maximum)
            for k in range(self.n_clusters, self.max_n_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=self.random_state)
                labels = kmeans.fit_predict(embeddings_np)
                score = silhouette_score(embeddings_np, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_model = kmeans
        else:
            best_k = self.n_clusters
            best_model = KMeans(n_clusters=best_k, random_state=self.random_state)
            best_model.fit(embeddings_np)
        
        assert best_model is not None, "Best model is None"
        buffer['clusters'] = best_model
        buffer['cluster_centers'] = best_model.cluster_centers_
        logger.info(
            f"Updated clusters for concept option '{concept_option}' with {len(buffer['texts'])} samples using K={best_k}"
        )
        self.random_state += 1

    def assign_cluster(self, concept_option: str, texts: List[str]) -> np.ndarray:
        """
        For a batch of texts, return a Long tensor of the assigned clusters, 
        where the i-th element is the cluster ID for the i-th text (between 0 and n_clusters-1).
        
        Args:
            concept_option: The concept option
            texts: The texts to assign clusters to
            
        Returns:
            Cluster IDs for the texts
        """
        if concept_option not in self.option_buffers:
            raise ValueError(f"Concept option '{concept_option}' not found")
            
        buffer = self.option_buffers[concept_option]
        
        if buffer['clusters'] is None:
            raise ValueError(f"KMeans has not been trained yet for '{concept_option}'")
            
        # Embed the texts, the output is already a 2D ndarray of size (num_inputs, output_dimension)
        embeddings = self.embedder.encode(texts)
            
        # Predict cluster
        labels = buffer['clusters'].predict(embeddings)
        
        return labels
        
    def get_n_clusters(self, concept_option: str) -> int:
        """
        Get the number of clusters for a specific concept option.
        
        Args:
            concept_option: The concept option
        """
        if concept_option not in self.option_buffers:
            raise ValueError(f"Concept option '{concept_option}' not found")
        return self.option_buffers[concept_option]['clusters'].n_clusters
        
    def get_stats(self) -> Dict[str, Dict]:
        """
        Get statistics about the buffer.
        
        Returns:
            Dictionary with buffer statistics
        """
        stats = {}
        
        for option, buffer in self.option_buffers.items():
            stats[option] = {
                'num_samples': len(buffer['texts']),
                'clustered': buffer['clusters'] is not None,
            }
            
            if buffer['clusters'] is not None:
                embeddings_np = buffer['embeddings']
                labels = buffer['clusters'].predict(embeddings_np)
                unique_labels, counts = np.unique(labels, return_counts=True)
                stats[option]['cluster_counts'] = dict(zip(unique_labels.tolist(), counts.tolist()))

                # add some sample text for each cluster in the current buffer
                stats[option]['cluster_samples'] = {i: None for i in range(buffer['clusters'].n_clusters)}
                for i in range(buffer['clusters'].n_clusters):
                    for j, text in enumerate(buffer['texts']):
                        if labels[j] == i:
                            stats[option]['cluster_samples'][i] = text
                            break
        return stats


if __name__ == "__main__":
    cache_dir = "/net/scratch2/listar2000/gfn-od/models/pretrained/sentence_transformer"
    sentence_transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=cache_dir)

    buffer = DiversityReplayBuffer(sentence_transformer, n_clusters=2, buffer_size=20, update_clusters_every=2, min_samples_for_clustering=3)

    print("Transformers device:", sentence_transformer.device)
    
    concept = "animal"
    initial_texts = ["Dogs are cute", "Cats are cute", "Dogs are friendly", "Kittens are friendly", "Puppies are cute"]
    buffer.add_samples(concept, initial_texts)
    
    animals = {"Dogs", "Cats", "Puppies", "Kittens"}
    adjectives = {"cute", "friendly"}

    stats = buffer.get_stats()

    import random
    for i in range(10):
        new_texts = [f"{random.choice(list(animals))} are {random.choice(list(adjectives))}" for _ in range(3)]
        assigned = buffer.add_samples(concept, new_texts)
        
        stats = buffer.get_stats()
        for i in range(len(new_texts)):
            print(str(assigned[i].item()) + " => " + new_texts[i])

        print(stats)

