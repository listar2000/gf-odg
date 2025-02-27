from state import Concept, ConceptBlock, OpenBlock, AbstractBlock
from typing import List, Optional
import collections


class RawTextProcessor:
    def __init__(self, concepts: List[Concept], max_window_size: int = 5):
        """Process a sequence of decoded tokens into blocks

        Args:
            concepts: List of Concepts to process
        """
        self.concepts = {concept.name: concept for concept in concepts}
        self.max_window_size = max_window_size

    def process_text_to_trajectory(self, raw_text: List[str]) -> List[AbstractBlock]:
        concepts_to_check = set(self.concepts.keys())
        blocks = []
        # track the index of the start of each block
        block_indices = []

        open_buffer = []
        window = collections.deque(maxlen=self.max_window_size)
        
        for i, token in enumerate(raw_text):
            # Check every suffix ending with the new token
            window.append(token)
            open_buffer.append(token)
            concept_block, L = self.detect_concepts(window, concepts_to_check)
            if concept_block is not None: # matched
                concepts_to_check.remove(concept_block.concept.name)
                window.clear()
                buffer_flushed = open_buffer[:-L]
                if buffer_flushed:
                    block_indices.append(i - len(open_buffer) + 1)
                    blocks.append(OpenBlock(0, buffer_flushed))
                blocks.append(concept_block)
                block_indices.append(i - L + 1)
                open_buffer = []

        # flush the buffer
        if open_buffer:
            block_indices.append(len(raw_text) - len(open_buffer))
            blocks.append(OpenBlock(0, open_buffer))

        return blocks, block_indices   

    def detect_concepts(self, window: collections.deque, concepts_to_check: set[str]) -> Optional[str]:
        window_list = list(window)
        for L in range(1, len(window_list) + 1):
            word = "".join(window_list[-L:]).strip()
            for concept_name in concepts_to_check:
                concept = self.concepts[concept_name]
                match_words = concept.get_match_words()
                if word in match_words:
                    matched_option = concept.alias_to_option[word]
                    concept_block = ConceptBlock(concept, matched_option, window_list[-L:])
                    # concepts_to_check.remove(concept_name)
                    # window.clear()
                    return concept_block, L
        return None, -1


if __name__ == "__main__":
    import time

    animal = Concept("animal", ["cat", "dog"], case_variants=["capitalized", "upper"])
    color = Concept("color", ["red", "blue"], case_variants=["capitalized", "upper"])
    raw_text = ["we", " love", " animal", " like", " cat", " since", " it", " is", " a", " red", "."] * 2
    processor = RawTextProcessor([animal, color], max_window_size=2)

    iterations = 10000
    start_time = time.time()

    for _ in range(iterations):
        processor.process_text_to_trajectory(raw_text)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Processing time for {iterations} iterations: {elapsed_time:.4f} seconds")

