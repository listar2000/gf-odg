import os
from typing import Dict

import hydra
import torch
from datasets import DatasetDict
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from src.data.data_loader import DataLoader as DatasetLoader
from src.models.sft import SFTModel
from src.training.sft_trainer import SFTTrainer


def prepare_dataset(
    dataset: DatasetDict,
    tokenizer,
    config: DictConfig,
    prompt_template: str,
    input_fields: Dict[str, str],
) -> Dict[str, DataLoader]:
    """Prepare dataset for training.

    Args:
        dataset: The dataset to prepare
        tokenizer: Tokenizer to use
        config: Training configuration
        prompt_template: Prompt template to use
        input_fields: Input fields to use

    Returns:
        Dictionary containing train and validation dataloaders
    """
    # Format dataset for instruction tuning (just Q&A pairs)
    dataset_loader = DatasetLoader(config)
    formatted_dataset = dataset_loader.format_for_instruction_tuning(
        dataset=dataset["train"],
        prompt_template=prompt_template,
        input_fields=input_fields,
        max_length=config.models.gemma_2b.sft.training.max_length,
    )

    print("Formatted dataset features:", formatted_dataset.features)
    print("First example:", formatted_dataset[0])

    # Create train/val split
    dataset_dict = formatted_dataset.train_test_split(test_size=0.1)

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=config.models.gemma_2b.sft.training.max_length,
            return_tensors=None  # Important: let the collator handle tensors
        )

    tokenized_datasets = DatasetDict({
        "train": dataset_dict["train"].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset_dict["train"].column_names
        ),
        "validation": dataset_dict["test"].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset_dict["test"].column_names
        ),
    })
    # Create dataloaders
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We want causal LM
    )

    return {
        "train": DataLoader(
            tokenized_datasets["train"],
            batch_size=config.models.gemma_2b.sft.training.batch_size,
            collate_fn=data_collator,
            shuffle=True,
        ),
        "validation": DataLoader(
            tokenized_datasets["validation"],
            batch_size=config.models.gemma_2b.sft.training.batch_size,
            collate_fn=data_collator,
        ),
    }


@hydra.main(config_path="../configs", config_name="main", version_base="1.1")
def main(config: DictConfig):
    """Main training function.

    Args:
        config: Hydra configuration
    """
    # Get model config (using gemma_2b for now)
    model_config = config.models.gemma_2b

    if not model_config.sft.enabled:
        raise ValueError("SFT is not enabled in the config")

    print("Configuration:")
    print(OmegaConf.to_yaml(config))

    # Load dataset
    dataset_loader = DatasetLoader(config)
    dataset_name = model_config.sft.data.train_dataset
    dataset = dataset_loader.load_dataset(dataset_name)

    # Get dataset-specific template
    if dataset_name not in config.datasets:
        raise ValueError(f"Dataset {dataset_name} not found in configuration")
    prompt_template = config.datasets[dataset_name].template

    # Define dataset-specific field mappings
    field_mappings = {
        "career_qa": {
            "question": "question",
            "answer": "answer"
        },
        "doqa": {
            "question": "question",
            "answer": "answers.text[0]",
            "domain": "title"
        }
    }

    if dataset_name not in field_mappings:
        raise ValueError(
            f"Field mappings not defined for dataset {dataset_name}")

    # Initialize model
    model = SFTModel(
        model_name=model_config.model_path,
        peft_config=model_config.sft.peft,
        use_flash_attention=model_config.model_kwargs.get(
            "use_flash_attention_2", True),
    )

    # Prepare datasets with dataset-specific template and fields
    dataloaders = prepare_dataset(
        dataset=dataset,
        tokenizer=model.get_tokenizer,
        config=config,
        prompt_template=prompt_template,
        input_fields=field_mappings[dataset_name]
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataloader=dataloaders["train"],
        val_dataloader=dataloaders["validation"],
        learning_rate=model_config.sft.training.learning_rate,
        warmup_steps=model_config.sft.training.warmup_steps,
        max_steps=model_config.sft.training.max_steps,
        gradient_accumulation_steps=model_config.sft.training.gradient_accumulation_steps,
        wandb_config=model_config.sft.logging.wandb,
    )

    # Train model
    print("Starting training...")
    metrics = trainer.train(
        checkpoint_dir=model_config.sft.output.checkpoint_dir)
    print("Training finished!")
    print(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()
