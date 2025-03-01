from state import Concept, ConceptBlock, OpenBlock, AbstractBlock
from typing import List, Optional
import collections


class RawTextProcessor:
    def __init__(self, concepts: List[Concept], max_window_size: int = 5, only_concepts: bool = False):
        """Process a sequence of decoded tokens into blocks

        Args:
            concepts: List of Concepts to process
        """
        self.concepts = {concept.name: concept for concept in concepts}
        self.max_window_size = max_window_size
        self.only_concepts = only_concepts

    def process_text_to_trajectory(self, raw_text: List[str]) -> List[AbstractBlock]:
        concepts_to_check = list(self.concepts.keys())  
        blocks = []
        block_indices = []

        open_buffer = []
        window = collections.deque(maxlen=self.max_window_size)
        buffer_start = 0  # Track the start of an OpenBlock

        for i, token in enumerate(raw_text):
            # Check every suffix ending with the new token
            window.append(token)
            open_buffer.append(token)

            for concept in concepts_to_check:  # Iterate in order
                concept_block, L = self.detect_concepts(window, [concept])
                if concept_block is not None:  # Matched
                    concepts_to_check.remove(concept) 
                    window.clear()
                    
                    if not self.only_concepts:  # Skip OpenBlocks if only_concepts is True
                        buffer_flushed = open_buffer[:-L]
                        if buffer_flushed:
                            block_indices.append((buffer_start, i - L))  # Store (start, end)
                            blocks.append(OpenBlock(0, buffer_flushed))

                    blocks.append(concept_block)
                    block_indices.append((i - L + 1, i + 1))  # Store (start, end)
                    open_buffer = []
                    buffer_start = i + 1  # Reset start index for next OpenBlock
                    break  # Exit the loop after detecting one concept

        # Flush the buffer if `only_concepts` is False
        if open_buffer and not self.only_concepts:
            block_indices.append((buffer_start, len(raw_text)))  # Store (start, end)
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

    ListOfFlowerNames = ["Cosmos", "Cornflower", "Dahlia", "Zinnia", "Chrysanthemum", "Celosia", "Larkspur", "Gladiolus", "Craspedia", "Gomphrena", "Sunflower", "Gerbera Daisy", "Snapdragon", "Bells of Ireland", "Stock", "Strawflower", "Nigella", "Nicotiana", "Nasturtium", "Petunia", "Marigold", "Impatiens", "Pansy", "Sweet Alyssum", "Morning Glory", "Coneflower", "Black-Eyed Susan", "Hosta", "Peony", "Daylily", "Lavender", "Phlox", "Shasta Daisy", "Bleeding Heart", "Iris", "Hellebore", "Yarrow", "Salvia", "Veronica", "Gaillardia", "Coreopsis", "Columbine", "Lupine", "Delphinium", "Astilbe", "Foxglove", "Hollyhock", "Sweet William", "Canterbury Bells", "Forget-Me-Not", "Evening Primrose", "Honesty", "Parsley", "Angelica", "Rose", "Tulip", "Orchid", "Lily", "Hydrangea", "Carnation", "Freesia", "Ranunculus", "Anemone", "Gardenia", "Azalea", "Camellia", "Jasmine", "Magnolia", "Bougainvillea"]
    
    N_Concepts = 4
    flowers = [Concept(f"flower{i+1}", ListOfFlowerNames, case_variants=["capitalized", "lower", "plural"]) for i in range(N_Concepts)]
    text_processor = RawTextProcessor(flowers, max_window_size=N_Concepts, only_concepts=True)
    raw_text = ["Ro","se",",", "Cornflower", "Dahlia", "Dahlia",","] 

    blocks, s = text_processor.process_text_to_trajectory(raw_text)
    #print type of blocks
    print(type(blocks))
    print(blocks)
    print(s)
