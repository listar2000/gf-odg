from typing import Dict, Optional
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import wandb
from omegaconf import OmegaConf

from src.models.sft import SFTModel


class SFTTrainer:
    """Trainer for Supervised Fine-Tuning."""

    def __init__(
        self,
        model: SFTModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        max_steps: int = 1000,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
        wandb_config: Optional[Dict] = None,
    ):
        """Initialize the trainer.

        Args:
            model: The SFT model to train
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            learning_rate: Learning rate for training
            warmup_steps: Number of warmup steps
            max_steps: Maximum number of training steps
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to train on
            wandb_config: Optional W&B configuration
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device

        # Training parameters
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.get_model.parameters(),
            lr=learning_rate,
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )

        # Convert OmegaConf to dict if needed
        if wandb_config and hasattr(wandb_config, "_content"):
            wandb_config = OmegaConf.to_container(wandb_config, resolve=True)

        # Initialize W&B if config is provided
        self.wandb_enabled = wandb_config is not None and wandb_config.get(
            "enabled", False)
        if self.wandb_enabled:
            # Format run name with timestamp
            run_name = wandb_config.get("name", "sft-run")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{run_name}_{timestamp}"

            # Initialize W&B
            wandb.init(
                project=wandb_config.get("project", "gemma-sft"),
                name=run_name,
                tags=wandb_config.get("tags", []),
                config={
                    "learning_rate": learning_rate,
                    "warmup_steps": warmup_steps,
                    "max_steps": max_steps,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "model_name": model.model_name,
                    "batch_size": train_dataloader.batch_size,
                    "device": device,
                }
            )

    def train(self, checkpoint_dir: Optional[str] = None) -> Dict[str, float]:
        """Train the model.

        Args:
            checkpoint_dir: Optional directory to save checkpoints

        Returns:
            Dictionary containing training metrics
        """
        self.model.get_model.train()
        total_loss = 0
        step = 0
        best_val_loss = float('inf')
        training_start_time = time.time()

        progress_bar = tqdm(total=self.max_steps, desc="Training")

        while step < self.max_steps:
            for batch in self.train_dataloader:
                if step >= self.max_steps:
                    break

                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    labels=batch["labels"].to(self.device),
                )
                loss = outputs["loss"] / self.gradient_accumulation_steps
                total_loss += loss.item()

                # Backward pass
                loss.backward()

                # Update weights if gradient accumulation is done
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.get_model.parameters(),
                        self.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    # Log metrics
                    metrics = {
                        "train/loss": total_loss / (step + 1),
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "train/step": step,
                        "train/epoch": step / len(self.train_dataloader),
                        "train/elapsed_time": time.time() - training_start_time,
                    }

                    # Evaluate if validation data is available
                    if self.val_dataloader is not None and step % 100 == 0:
                        val_loss = self.evaluate()
                        metrics["val/loss"] = val_loss

                        # Save best model
                        if checkpoint_dir and val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self.model.save(f"{checkpoint_dir}/best_model")
                            metrics["val/best_loss"] = best_val_loss

                        self.model.get_model.train()
                        progress_bar.set_postfix({
                            "train_loss": total_loss / (step + 1),
                            "val_loss": val_loss,
                        })
                    else:
                        progress_bar.set_postfix({
                            "train_loss": total_loss / (step + 1),
                        })

                    # Log to W&B
                    if self.wandb_enabled:
                        wandb.log(metrics, step=step)

                # Save checkpoint if directory is provided
                if checkpoint_dir and step % 200 == 0:
                    self.model.save(f"{checkpoint_dir}/checkpoint-{step}")

                step += 1
                progress_bar.update(1)

        progress_bar.close()

        # Final metrics
        final_metrics = {
            "train_loss": total_loss / step,
            "best_val_loss": best_val_loss if self.val_dataloader else None,
            "total_steps": step,
            "training_time": time.time() - training_start_time,
        }

        if self.wandb_enabled:
            wandb.log(final_metrics)
            wandb.finish()

        return final_metrics

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate the model on validation data.

        Returns:
            Average validation loss
        """
        if self.val_dataloader is None:
            return 0.0

        self.model.get_model.eval()
        total_loss = 0
        num_batches = 0

        for batch in self.val_dataloader:
            outputs = self.model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                labels=batch["labels"].to(self.device),
            )
            total_loss += outputs["loss"].item()
            num_batches += 1

        return total_loss / num_batches
