from typing import Dict, Optional
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import wandb
from omegaconf import OmegaConf

from src.models.gflownet import GFlowNetModel
from gfn.losses import trajectory_balance_loss


class GFlowNetTrainer:
    """Trainer for GFlowNet-based Reinforcement Learning without a replay buffer."""

    def __init__(
        self,
        model: GFlowNetModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        warmup_steps: int = 100,
        max_steps: int = 1000,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
        wandb_config: Optional[Dict] = None,
    ):
        """Initialize the GFlowNet trainer."""
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

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            list(self.model.get_model.parameters()) + [self.model.logZ],
            lr=learning_rate,
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
        )

        # Convert OmegaConf to dict if needed
        if wandb_config and hasattr(wandb_config, "_content"):
            wandb_config = OmegaConf.to_container(wandb_config, resolve=True)

        # Initialize W&B if config is provided
        self.wandb_enabled = wandb_config is not None and wandb_config.get("enabled", False)
        if self.wandb_enabled:
            run_name = wandb_config.get("name", "gfn-run")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{run_name}_{timestamp}"

            wandb.init(
                project=wandb_config.get("project", "gflownet-training"),
                name=run_name,
                tags=wandb_config.get("tags", []),
                config={
                    "learning_rate": learning_rate,
                    "warmup_steps": warmup_steps,
                    "max_steps": max_steps,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "model_name": model.get_model.config._name_or_path,
                    "batch_size": train_dataloader.batch_size,
                    "device": device,
                }
            )

    def train(self, checkpoint_dir: Optional[str] = None) -> Dict[str, float]:
        """Train the GFlowNet model."""
        self.model.get_model.train()
        total_loss = 0
        step = 0
        best_val_loss = float("inf")
        training_start_time = time.time()

        progress_bar = tqdm(total=self.max_steps, desc="Training")

        while step < self.max_steps:
            for batch in self.train_dataloader:
                if step >= self.max_steps:
                    break

                prompt, goal = batch["initial_state"], batch["goal"]

                # Sample fresh trajectory (autoregressive)
                actions, states, log_pf = self.model.sample_trajectory(prompt)

                # Compute reward from final state
                reward = self.model.compute_reward(states, goal)
                log_r = torch.log(reward + 1e-4)

                # Compute trajectory balance loss
                loss = trajectory_balance_loss(log_pf, log_r, self.model.logZ)
                total_loss += loss.item() / self.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Update weights if gradient accumulation is done
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.get_model.parameters()) + [self.model.logZ],
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

                # Save checkpoint
                if checkpoint_dir and (step + 1) % 200 == 0:
                    self.model.save(f"{checkpoint_dir}/checkpoint-{step + 1}")

                step += 1
                progress_bar.update(1)

        progress_bar.close()

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
