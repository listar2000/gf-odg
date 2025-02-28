import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
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
            buffer_size: Maximum size of each buffer
            update_clusters_every: Update clusters after this many additions
            min_samples_for_clustering: Minimum number of samples required before clustering
        """
        self.embedder = embedder
        self.n_clusters = n_clusters
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
        force_update: bool = False
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
            
        # Trim buffer if it exceeds maximum size
        if len(buffer['texts']) > self.buffer_size:
            buffer['texts'] = buffer['texts'][-self.buffer_size:]
            buffer['embeddings'] = buffer['embeddings'][-self.buffer_size:]
            
        # Update counter
        buffer['update_counter'] += len(texts)
        
        # Update clusters if needed
        if (buffer['update_counter'] >= self.update_clusters_every and 
            len(buffer['texts']) >= self.min_samples_for_clustering) or force_update:
            self._update_clusters(concept_option)
            buffer['update_counter'] = 0

        if buffer['clusters'] is not None:
            assigned = self.assign_cluster(concept_option, texts)
            return assigned
        else:
            return None
            
    def _update_clusters(self, concept_option: str, force_update: bool = False) -> None:
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
        
        # Fit KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        kmeans.fit(embeddings_np)
        
        # Store clusters and centers
        buffer['clusters'] = kmeans
        buffer['cluster_centers'] = kmeans.cluster_centers_
        
        self.random_state += 1 # For next time
        logger.info(f"Updated clusters for concept option '{concept_option}' with {len(buffer['texts'])} samples")


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
        
    def get_cluster_centers(self, concept_option: str) -> Optional[np.ndarray]:
        """
        Get cluster centers for a specific concept option.
        
        Args:
            concept_option: The concept option
            
        Returns:
            Numpy array of cluster centers or None if not available
        """
        if concept_option not in self.option_buffers:
            return None
            
        return self.option_buffers[concept_option].get('cluster_centers')
        
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

