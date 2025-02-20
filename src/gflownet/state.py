from typing import List, Dict, Set
import abc
import torch

CASE_VARIANTS = ["capitalized", "upper", "lower", "plural"]

class Concept:
    def __init__(self, name: str, options: List[str], aliases: Dict[str, Set[str]] = None, case_variants: List[str] = None):
        """
        Initialize a concept with a name and a finite list of valid options.
        :param name: Name of the concept, e.g. 'animal'
        :param options: List of valid tokens, e.g. ['cat', 'dog']
        """
        self.name = name
        self.options: Set[str] = set(options)

        if aliases is not None:
            assert all([k in self.options for k in aliases.keys()]), \
                f"Invalid aliases for option: {k}"
        self.aliases: Dict[str, Set[str]] = aliases or {option: set() for option in self.options}

        if case_variants is not None:
            assert all([v in CASE_VARIANTS for v in case_variants]), \
                f"Invalid case variants: {case_variants}"
        self.case_variants: List[str] = case_variants

        # get reverse alias -> option mapping
        self.alias_to_option: Dict[str, str] = {}

        # insert option into its own aliases
        for option in self.options:
            self.aliases[option].add(option)

        for option, aliases in self.aliases.items():
            for alias in aliases:
                assert alias not in self.alias_to_option, "identical aliases across different options"
                self.alias_to_option[alias] = option
                if "plural" in self.case_variants:
                    self.alias_to_option[alias + "s"] = option
                if "capitalized" in self.case_variants:
                    self.alias_to_option[alias.capitalize()] = option
                    if "plural" in self.case_variants:
                        self.alias_to_option[alias.capitalize() + "s"] = option
                if "upper" in self.case_variants:
                    self.alias_to_option[alias.upper()] = option
                    if "plural" in self.case_variants:
                        self.alias_to_option[alias.upper() + "s"] = option
                if "lower" in self.case_variants:
                    self.alias_to_option[alias.lower()] = option
                    if "plural" in self.case_variants:
                        self.alias_to_option[alias.lower() + "s"] = option

        self.match_words: Set[str] = set(self.alias_to_option.keys())
    
    def get_match_words(self):
        return self.match_words

    def __repr__(self):
        return f"Concept(name={self.name}, options={list(self.options)})"


class AbstractBlock(abc.ABC):
    def __init__(self):
        self.raw_text = ""
        self.prob = None

    def get_raw_text(self) -> str:
        return [""]

    def is_start_block(self) -> bool:
        return False

    def is_end_block(self) -> bool:
        return False


class StartBlock(AbstractBlock):
    def __init__(self):
        super().__init__()

    def is_start_block(self) -> bool:
        return True

    def __repr__(self):
        return "<StartBlock>"


class EndBlock(AbstractBlock):
    def __init__(self):
        super().__init__()

    def is_end_block(self) -> bool:
        return True

    def __repr__(self):
        return "<EndBlock>"


class ConceptBlock(AbstractBlock):
    def __init__(self, concept: Concept, option: str, raw_text: List[str]):
        super().__init__()
        self.concept = concept

        assert option in self.concept.options, f"Invalid option: {option}"
        self.option = option

        # make sure the raw_text is indeed in the option's aliases
        # also note that we store the raw_text with spaces retained, so strip when checking
        # assert ''.join(raw_text).strip() in self.concept.aliases[option], f"Invalid raw_text: {raw_text}"
        self.raw_text = raw_text

    def get_raw_text(self) -> List[str]:
        return self.raw_text

    def __repr__(self):
        return f"<ConceptBlock {self.concept.name}:{self.option}>"
        

class OpenBlock(AbstractBlock):
    def __init__(self, encoded: torch.Tensor, raw_text: List[str]):
        super().__init__()
        self.encoded = encoded
        self.raw_text = raw_text

    def get_raw_text(self) -> List[str]:
        return self.raw_text

    def __repr__(self):
        text = "".join(self.raw_text)
        if len(text) > 10:
            text = text[:7] + "..."
        return f"<OpenBlock {text}>"


class State:
    def __init__(self, blocks: List[AbstractBlock]):
        self.blocks = blocks
        self.step = len(blocks)

        # self.concept_blocks, self.open_blocks = [], []
        self.concepts = {}
        for block in self.blocks:
            if isinstance(block, ConceptBlock):
                self.concepts[block.concept.name] = block.option
        
        self.is_end_state = isinstance(self.blocks[-1], EndBlock)
        self.concept_next = isinstance(self.blocks[-1], OpenBlock)
        self.open_next = isinstance(self.blocks[-1], ConceptBlock)

    def get_raw_text(self) -> str:
        full_text_list = []
        for block in self.blocks:
            full_text_list += block.get_raw_text()
        return "".join(full_text_list)


class Trajectory:
    def __init__(self, blocks: List[AbstractBlock]):
        self.blocks = blocks
        assert self.blocks[0].is_start_block(), "First block must be a start block"
        assert self.blocks[-1].is_end_block(), "Last block must be an end block"

        # check that open and concept blocks are interleaved
        prev_block = self.blocks[1]
        for block in self.blocks[2:-1]:
            if isinstance(prev_block, OpenBlock):
                assert isinstance(block, ConceptBlock), "Expected concept block, got: " + str(block)
            elif isinstance(prev_block, ConceptBlock):
                assert isinstance(block, OpenBlock), "Expected open block, got: " + str(block)
            prev_block = block

    def get_state_at_step(self, step: int):
        # step must be in [0, len(blocks) - 1]
        if step >= len(self.blocks):
            return None

        return State(self.blocks[:step + 1])

if __name__ == "__main__":
    aliases = {"cat": {"catty", "kitten"}, "dog": {"doggy", "puppy"}}
    animal = Concept("animal", ["cat", "dog"], aliases=aliases, case_variants=["capitalized", "upper"])

    animal_block = ConceptBlock(animal, "cat", ["cat"])
    open_block = OpenBlock(torch.zeros(1), [" is", " an", " animal", "."])
    state = State([animal_block, open_block])
    print(state.get_raw_text())