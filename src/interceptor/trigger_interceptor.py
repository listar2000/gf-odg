import collections
from typing import List, Optional, Dict, Any, Union

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from src.interceptor.base_interceptor import BaseInterceptor, InterceptionResult

############################
# Trie implementation
############################


class TrieNode:
    def __init__(self):
        self.children = {}  # Maps token -> TrieNode
        self.is_end = False  # True if a complete trigger sequence ends here
        self.trigger_word = None  # Optionally, store the trigger word (or id)


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, token_sequence: List[int], trigger_word: Optional[str] = None):
        """
        Insert a token sequence (list of token ids) into the trie.
        Optionally store a string representation of the trigger.
        """
        node = self.root
        for token in token_sequence:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.is_end = True
        node.trigger_word = trigger_word

############################
# Sliding Window Trigger Detector with EOS Handling
############################


class SlidingWindowTriggerDetector:
    def __init__(self, trie: Trie, batch_size: int, max_trigger_length: int, eos_token: int):
        """
        :param trie: A Trie containing trigger token sequences.
        :param batch_size: Number of parallel sequences (batch size).
        :param max_trigger_length: Maximum length (in tokens) of any trigger.
        :param eos_token: The token id that signals the end of a sequence.
        """
        self.trie = trie
        self.batch_size = batch_size
        self.max_trigger_length = max_trigger_length
        self.eos_token = eos_token
        # For each sample, maintain a sliding window (deque) of the last n tokens.
        self.windows = [collections.deque(
            maxlen=max_trigger_length) for _ in range(batch_size)]
        # Maintain a flag for each sample to indicate if the sequence has terminated.
        self.finished = [False for _ in range(batch_size)]

    def process_batch(self, new_tokens: List[int]) -> List[Optional[str]]:
        """
        Process a batch of new tokens (one token per sample).

        :param new_tokens: List of length batch_size, each an int token.
        :return: List of length batch_size. For each sample, if a trigger was completed,
                 the corresponding trigger_word is returned; otherwise, None.
        """
        if len(new_tokens) != self.batch_size:
            raise ValueError("new_tokens length must equal batch_size")

        triggers_found = [None] * self.batch_size  # One per sample

        for i, token in enumerate(new_tokens):
            # If the sequence is already finished, skip processing.
            if self.finished[i]:
                continue

            # Check for EOS token.
            if token == self.eos_token:
                self.finished[i] = True
                # Optionally, you could clear the window or take other actions.
                self.windows[i].clear()
                continue

            window = self.windows[i]
            window.append(token)
            found_trigger = None

            # Check every suffix ending with the new token.
            window_list = list(window)  # Convert deque to list for slicing.
            for L in range(1, len(window_list) + 1):
                candidate = window_list[-L:]  # Suffix of length L.
                node = self.trie.root
                valid = True
                for t in candidate:
                    if t not in node.children:
                        valid = False
                        break
                    node = node.children[t]
                if valid and node.is_end:
                    found_trigger = node.trigger_word
                    # Optionally, clear the window upon detection.
                    window.clear()
                    break  # Stop checking further suffixes for this sample.

            triggers_found[i] = found_trigger

        return triggers_found

    def reset(self):
        """
        Reset the sliding window and finished flags for all samples.
        """
        self.windows = [collections.deque(
            maxlen=self.max_trigger_length) for _ in range(self.batch_size)]
        self.finished = [False for _ in range(self.batch_size)]

############################
# Decode-based Trigger Detector
############################


class SlidingWindowDecodeDetector:
    def __init__(self, trigger_words: set[str], batch_size: int, max_window_size: int, eos_token: int):
        """Initialize the decode-based sliding window detector.

        Args:
            trigger_words: Set of trigger words to detect (all lowercase)
            batch_size: Number of parallel sequences
            max_window_size: Maximum number of tokens to keep in sliding window
            eos_token: Token ID that signals end of sequence
        """
        self.trigger_words = trigger_words
        self.batch_size = batch_size
        self.max_window_size = max_window_size
        self.eos_token = eos_token
        # For each sample, maintain a sliding window of tokens
        self.windows = [collections.deque(
            maxlen=max_window_size) for _ in range(batch_size)]
        # Track finished sequences
        self.finished = [False for _ in range(batch_size)]

    def process_batch(self, new_tokens: List[int], tokenizer) -> List[Optional[str]]:
        """Process a batch of new tokens and check for triggers in decoded text.

        Args:
            new_tokens: List of length batch_size, each an int token
            tokenizer: Tokenizer to decode the tokens

        Returns:
            List of length batch_size. For each sample, returns the trigger word if found,
            otherwise None.
        """
        if len(new_tokens) != self.batch_size:
            raise ValueError("new_tokens length must equal batch_size")

        triggers_found = [None] * self.batch_size

        for i, token in enumerate(new_tokens):
            # Skip if sequence is finished
            if self.finished[i]:
                continue

            # Check for EOS token
            if token == self.eos_token:
                self.finished[i] = True
                self.windows[i].clear()
                continue

            window = self.windows[i]
            window.append(token)

            # Check every suffix ending with the new token
            window_list = list(window)
            for L in range(1, len(window_list) + 1):
                # Get suffix of length L
                suffix = window_list[-L:]
                # Decode and normalize the suffix
                suffix_text = tokenizer.decode(suffix).strip().lower()

                if suffix_text in self.trigger_words:
                    triggers_found[i] = suffix_text
                    window.clear()
                    break

        return triggers_found

    def reset(self):
        """Reset the sliding windows and finished flags for all samples."""
        self.windows = [collections.deque(maxlen=self.max_window_size)
                        for _ in range(self.batch_size)]
        self.finished = [False for _ in range(self.batch_size)]


class DecodeInterceptor(BaseInterceptor):
    """Interceptor that detects trigger words by decoding token windows to text."""

    def __init__(self, tokenizer: AutoTokenizer, config: DictConfig):
        """Initialize the DecodeInterceptor.

        Args:
            tokenizer: Tokenizer used by the model
            config: Configuration containing trigger words and behavior settings.
                   Expected to have trigger_interceptor.trigger_words,
                   trigger_interceptor.enabled, and trigger_interceptor.stop_on_trigger
        """
        super().__init__(tokenizer, config)

    def _initialize_state(self, config: DictConfig) -> None:
        """Initialize interceptor state including trigger words and detector.

        Args:
            config: Configuration containing trigger words and settings
        """
        super()._initialize_state(config)

        trigger_config = config.trigger_interceptor
        case_sensitive = trigger_config.get("case_sensitive", False)

        # Process trigger words based on case sensitivity
        self.trigger_words = set()
        for trigger in trigger_config.trigger_words:
            if case_sensitive:
                self.trigger_words.add(trigger)
            else:
                # Get all case variants based on config
                variants = trigger_config.get(
                    "case_variants", ["original", "capitalized", "upper", "lower"])
                trigger_variants = self._get_case_variants(trigger, variants)
                # Store all variants in lowercase for case-insensitive matching
                self.trigger_words.update(v.lower() for v in trigger_variants)

        # Set a reasonable window size (can be configured if needed)
        self.max_window_size = trigger_config.get("max_window_size", 5)
        self.detector = None

    def _get_case_variants(self, word: str, variants: List[str]) -> List[str]:
        """Get different case variants of a word based on specified variant types.

        Args:
            word: The original trigger word
            variants: List of variant types to generate ("original", "capitalized", "upper", "lower")

        Returns:
            List of strings with different case variants
        """
        result = set()

        for variant in variants:
            if variant == "original":
                result.add(word)
            elif variant == "capitalized":
                result.add(word.capitalize())
            elif variant == "upper":
                result.add(word.upper())
            elif variant == "lower":
                result.add(word.lower())

        return list(result)

    def intercept(
        self,
        new_tokens: torch.Tensor,
        full_sequence: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[InterceptionResult, List[InterceptionResult]]:
        """Intercept and analyze newly generated tokens for trigger words.

        Args:
            new_tokens: Tensor of shape (batch_size, new_seq_len) containing newly
                generated tokens
            full_sequence: Optional tensor of shape (batch_size, full_seq_len) containing
                the full sequence including the new tokens
            **kwargs: Additional arguments (unused)

        Returns:
            List of InterceptionResults, one per sequence in the batch
        """
        if not self.is_active:
            return [InterceptionResult() for _ in range(new_tokens.shape[0])]

        batch_size = new_tokens.shape[0]

        if not self._batch_size:
            self._batch_size = batch_size
        elif self._batch_size != batch_size:
            raise ValueError(
                f"Batch size changed from {self._batch_size} to {batch_size}.")

        # Initialize or reinitialize detector if batch size changes
        if self.detector is None:
            self.detector = SlidingWindowDecodeDetector(
                trigger_words=self.trigger_words,
                batch_size=self._batch_size,
                max_window_size=self.max_window_size,
                eos_token=self.tokenizer.eos_token_id
            )

        # Process the last token for each sequence in the batch
        last_tokens = new_tokens[:, -1].tolist()
        triggers = self.detector.process_batch(last_tokens, self.tokenizer)

        # Create results
        results = []
        for trigger in triggers:
            if trigger is not None:
                result = InterceptionResult(
                    should_stop=self.config.trigger_interceptor.stop_on_trigger,
                    metadata={"trigger_word": trigger}
                )
            else:
                result = InterceptionResult()
            results.append(result)

        return results

    def reset(self):
        """Reset the sliding windows and finished flags for all samples."""
        if not self.detector:
            raise ValueError("Detector not initialized.")
        self.detector.reset()
############################
# Trigger Interceptor
############################


class TriggerInterceptor(BaseInterceptor):
    """Interceptor that detects trigger words in generated text using a sliding window approach."""

    def __init__(self, tokenizer: AutoTokenizer, config: DictConfig):
        """Initialize the TriggerInterceptor.

        Args:
            tokenizer: Tokenizer used by the model
            config: Configuration containing trigger words and behavior settings.
                   Expected to have trigger_interceptor.trigger_words,
                   trigger_interceptor.enabled, and trigger_interceptor.stop_on_trigger
        """
        super().__init__(tokenizer, config)

    def _get_case_variants(self, word: str, variants: List[str]) -> List[str]:
        """Get different case variants of a word based on specified variant types.

        Args:
            word: The original trigger word
            variants: List of variant types to generate ("original", "capitalized", "upper", "lower")

        Returns:
            List of strings with different case variants
        """
        result = set()  # Use set to avoid duplicates

        for variant in variants:
            if variant == "original":
                result.add(word)
            elif variant == "capitalized":
                result.add(word.capitalize())
            elif variant == "upper":
                result.add(word.upper())
            elif variant == "lower":
                result.add(word.lower())

        return list(result)

    def _initialize_state(self, config: DictConfig) -> None:
        """Initialize interceptor state including the trie and detector.

        Args:
            config: Configuration containing trigger words and settings
        """
        super()._initialize_state(config)

        # Build trie from tokenized trigger words
        self.trie = Trie()
        self._max_trigger_length = 0

        trigger_config = config.trigger_interceptor
        case_sensitive = trigger_config.get("case_sensitive", False)

        for trigger in trigger_config.trigger_words:
            # If case-sensitive, just add the original trigger
            if case_sensitive:
                tokens = self.tokenizer.encode(
                    trigger, add_special_tokens=False)
                self._max_trigger_length = max(
                    self._max_trigger_length, len(tokens))
                self.trie.insert(tokens, trigger)
            else:
                # Get all case variants based on config
                variants = trigger_config.get(
                    "case_variants", ["original", "capitalized", "upper", "lower"])
                trigger_variants = self._get_case_variants(trigger, variants)

                # Add each variant to the trie
                for variant in trigger_variants:
                    tokens = self.tokenizer.encode(
                        variant, add_special_tokens=False)
                    self._max_trigger_length = max(
                        self._max_trigger_length, len(tokens))
                    # Store the original trigger word as the trigger_word for all variants
                    self.trie.insert(tokens, variant)

        # Initialize detector (will be created when batch size is known)
        self.detector = None
        self.stop_on_trigger = trigger_config.get("stop_on_trigger", True)

    def intercept(
        self,
        new_tokens: torch.Tensor,
        full_sequence: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[InterceptionResult, List[InterceptionResult]]:
        """Intercept and analyze newly generated tokens for trigger words.

        Args:
            new_tokens: Tensor of shape (batch_size, new_seq_len) containing newly
                generated tokens
            full_sequence: Optional tensor of shape (batch_size, full_seq_len) containing
                the full sequence including the new tokens
            **kwargs: Additional arguments (unused)

        Returns:
            List of InterceptionResults, one per sequence in the batch
        """
        if not self.is_active:
            return [InterceptionResult() for _ in range(new_tokens.shape[0])]

        batch_size = new_tokens.shape[0]

        # Initialize or reinitialize detector if batch size changes
        if self.detector is None or self.detector.batch_size != batch_size:
            self.detector = SlidingWindowTriggerDetector(
                trie=self.trie,
                batch_size=batch_size,
                max_trigger_length=self._max_trigger_length,
                eos_token=self.tokenizer.eos_token_id
            )

        # Process only the last token for each sequence in the batch
        # Since the sliding window maintains state, we only need to check the newest token
        # Get the last token for each batch
        last_tokens = new_tokens[:, -1].tolist()
        triggers = self.detector.process_batch(last_tokens)

        # Convert triggers to InterceptionResults
        results = []
        for trigger in triggers:
            result = InterceptionResult(
                should_stop=self.stop_on_trigger and trigger is not None,
                metadata={"trigger_word": trigger} if trigger else None
            )
            results.append(result)

        return results

    def reset(self) -> None:
        """Reset the interceptor's internal state."""
        if self.detector is not None:
            self.detector = None  # Will be recreated with next intercept call
