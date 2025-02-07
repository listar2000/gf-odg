from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer

from omegaconf import DictConfig


@dataclass
class InterceptionResult:
    """Result of an interception operation.

    Attributes:
        should_stop (bool): Whether the generation should be stopped
        edited_tokens (Optional[Union[torch.Tensor, List[int]]]): Modified tokens if any editing is needed
        metadata (Dict[str, Any]): Additional information about the interception
    """
    should_stop: bool = False
    edited_tokens: Optional[Union[torch.Tensor, List[int]]] = None
    metadata: Dict[str, Any] = None


class BaseInterceptor(ABC):
    """Base class for all interceptors.

    This class defines the interface that all interceptors must implement. Interceptors
    are used to monitor and potentially modify the token generation process of language
    models in real-time.
    """

    def __init__(self, tokenizer: AutoTokenizer, config: DictConfig):
        """Initialize the interceptor with configuration.

        Args:
            config: Dictionary containing configuration parameters
        """
        self.tokenizer = tokenizer
        self.config = config
        self._initialize_state(config)

    @abstractmethod
    def _initialize_state(self, config: DictConfig) -> None:
        """Initialize or reset any internal state needed by the interceptor."""
        if hasattr(config, "enabled") and not config.enabled:
            self.is_active = False
        else:
            self.is_active = True

        self._batch_size = None

    @abstractmethod
    def intercept(
        self,
        new_tokens: torch.Tensor,
        full_sequence: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[InterceptionResult, List[InterceptionResult]]:
        """Intercept and analyze newly generated tokens.

        Args:
            new_tokens: Tensor of shape (batch_size, new_seq_len) containing newly
                generated tokens
            full_sequence: Optional tensor of shape (batch_size, full_seq_len) containing
                the full sequence including the new tokens
            **kwargs: Additional arguments that may be needed by specific implementations

        Returns:
            A single InterceptionResult or a list of results (one per sequence in batch)
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the interceptor's internal state.

        This should be called between different generation sessions.
        """
        self._initialize_state()

    def enable(self) -> None:
        """Enable the interceptor."""
        self.is_active = True

    def disable(self) -> None:
        """Disable the interceptor."""
        self.is_active = False
