import os
import json
import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from tqdm import tqdm
from typing import Dict, List, Tuple

from src.models.gflownet import GFlowNetModel
from src.training.gflownet_trainer import GFlowNetTrainer


class Game24Dataset(Dataset):
    """Custom Dataset for Game24 using 24.csv and train.json"""

    def __init__(self, data_path: str, tokenizer, prompt_template: str, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template

        # Load and process dataset
        self.samples = self._load_data(data_path)

    def _load_data(self, data_path: str) -> List[Dict]:
        """Load dataset from CSV and JSON files."""
        csv_path = os.path.join(data_path, "24.csv")
        json_path = os.path.join(data_path, "train.json")

        with open(json_path, "r") as f:
            json_data = json.load(f)

        csv_data = []
        with open(csv_path, "r") as f:
            for line in f.readlines()[1:]:  # Skip header
                parts = line.strip().split(",")
                puzzle = " ".join(parts[1:5])
                solution = json_data.get(puzzle, "")  # Get solution from JSON if exists
                csv_data.append({"input": puzzle, "output": solution})

        return csv_data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_text = self.prompt_template.format(input=sample["input"])
        tokenized_sample = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokenized_sample["labels"] = self.tokenizer(
            sample["output"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        return {key: val.squeeze(0) for key, val in tokenized_sample.items()}


@hydra.main(config_path="../configs", config_name="main", version_base="1.1")
def main(config: DictConfig):
    """Main function for training GFlowNet on Game24."""
    
    # Load model and tokenizer
    model_name = config.models.gemma_2b.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = GFlowNetModel(
        model_name=model_name,
        peft_config=config.models.gemma_2b.gflownet.peft,
        reward_temperature=config.models.gemma_2b.gflownet.training.reward_temperature,
        epsilon=config.models.gemma_2b.gflownet.training.epsilon,
        pf_temperature=config.models.gemma_2b.gflownet.training.pf_temperature,
    )

    # Load dataset
    dataset_path = config.models.gemma_2b.gflownet.data.train_dataset
    prompt_template = config.datasets["game24"].template

    train_dataset = Game24Dataset(dataset_path, tokenizer, prompt_template)
    val_dataset = Game24Dataset(dataset_path, tokenizer, prompt_template)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model.get_model, padding=True)

    train_loader = DataLoader(
        train_dataset, batch_size=config.models.gemma_2b.gflownet.training.batch_size, collate_fn=data_collator, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.models.gemma_2b.gflownet.training.batch_size, collate_fn=data_collator
    )

    # Initialize trainer
    trainer = GFlowNetTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=config.models.gemma_2b.gflownet.training.learning_rate,
        warmup_steps=config.models.gemma_2b.gflownet.training.warmup_steps,
        max_steps=config.models.gemma_2b.gflownet.training.max_steps,
        gradient_accumulation_steps=config.models.gemma_2b.gflownet.training.gradient_accumulation_steps,
        wandb_config=config.models.gemma_2b.gflownet.logging.wandb,
    )

    # Train model
    print("Starting GFlowNet training on Game24...")
    metrics = trainer.train(checkpoint_dir=config.models.gemma_2b.gflownet.output.checkpoint_dir)
    print("Training finished!")
    print(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()
