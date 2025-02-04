from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class BaseModel(ABC):
    """Abstract base class for all models in the project.
    
    This class defines the interface that all model implementations must follow.
    It ensures consistency across different model types (SFT, PPO, GFlowNet, etc.)
    and provides common functionality.
    """
    
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Optional labels for training [batch_size, seq_len]
            
        Returns:
            Dictionary containing model outputs (e.g., loss, logits)
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 512,
        **kwargs,
    ) -> torch.Tensor:
        """Generate text using the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            max_length: Maximum length of generated sequence
            **kwargs: Additional generation arguments
            
        Returns:
            Generated token IDs [batch_size, seq_len]
        """
        pass
    
    @property
    @abstractmethod
    def get_model(self) -> PreTrainedModel:
        """Get the underlying model.
        
        Returns:
            The actual model implementation
        """
        pass
    
    @property
    @abstractmethod
    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer.
        
        Returns:
            The tokenizer used by the model
        """
        pass
    
    def save(self, path: str) -> None:
        """Save the model and tokenizer.
        
        Args:
            path: Path to save the model to
        """
        self.get_model.save_pretrained(path)
        self.get_tokenizer.save_pretrained(path)
    
    def load(self, path: str) -> None:
        """Load the model and tokenizer.
        
        Args:
            path: Path to load the model from
        """
        self.get_model.from_pretrained(path)
        self.get_tokenizer.from_pretrained(path)
    
    @property
    def device(self) -> torch.device:
        """Get the device the model is on.
        
        Returns:
            The device (CPU/GPU) the model is currently on
        """
        return next(self.get_model.parameters()).device
    
    def to(self, device: torch.device) -> 'BaseModel':
        """Move the model to the specified device.
        
        Args:
            device: Device to move the model to
            
        Returns:
            Self for method chaining
        """
        self.get_model.to(device)
        return self
    
    def train(self, mode: bool = True) -> 'BaseModel':
        """Set the model to training mode.
        
        Args:
            mode: Whether to set training mode (True) or evaluation mode (False)
            
        Returns:
            Self for method chaining
        """
        self.get_model.train(mode)
        return self
    
    def eval(self) -> 'BaseModel':
        """Set the model to evaluation mode.
        
        Returns:
            Self for method chaining
        """
        return self.train(False)