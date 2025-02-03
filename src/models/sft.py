from typing import Dict, Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from omegaconf import OmegaConf

from .base_model import BaseModel


class SFTModel(BaseModel):
    """Supervised Fine-Tuning model with PEFT support."""

    def __init__(
        self,
        model_name: str,
        peft_config: Dict,
        use_flash_attention: bool = True,
    ):
        """Initialize the SFT model.

        Args:
            model_name: Path to pretrained model
            peft_config: PEFT configuration
            use_flash_attention: Whether to use flash attention
        """
        super().__init__()

        self.model_name = model_name
        # Convert OmegaConf to dict if needed
        if hasattr(peft_config, "_content"):
            peft_config = OmegaConf.to_container(peft_config, resolve=True)

        # Initialize model with eager attention
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # Configure PEFT (remove method field and add task_type)
        lora_config = {
            "r": peft_config["r"],
            "lora_alpha": peft_config["alpha"],
            "target_modules": peft_config["target_modules"],
            "lora_dropout": peft_config["dropout"],
            "bias": peft_config["bias"],
            "task_type": TaskType.CAUSAL_LM,
        }

        # Get PEFT model
        self._model = get_peft_model(self._model, LoraConfig(**lora_config))
        self._model.print_trainable_parameters()

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

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
            Dictionary containing model outputs (loss, logits)
        """
        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.forward(input_ids, attention_mask, labels)

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
        return self._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            pad_token_id=self._tokenizer.pad_token_id,
            **kwargs,
        )

    @property
    def get_model(self) -> PreTrainedModel:
        """Get the underlying model.

        Returns:
            The actual model implementation
        """
        return self._model

    @property
    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer.

        Returns:
            The tokenizer used by the model
        """
        return self._tokenizer
