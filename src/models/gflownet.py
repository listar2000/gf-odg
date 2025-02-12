from typing import Dict, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from torch.distributions import Categorical

from .base_model import BaseModel


class GFlowNetLLM(BaseModel):
    """GFlowNet model using an LLM as the policy function."""

    def __init__(
        self,
        model_name: str,
        reward_temperature: float = 1.0,
        epsilon: float = 0.1,
        pf_temperature: float = 1.0,
    ):
        """
        Initialize the GFlowNet model using an LLM as the policy.

        Args:
            model_name: Path to pretrained model.
            reward_temperature: Temperature scaling for reward function.
            epsilon: Epsilon-greedy exploration probability.
            pf_temperature: Temperature for policy function.
        """
        super().__init__()

        self.reward_temperature = reward_temperature
        self.epsilon = epsilon
        self.pf_temperature = pf_temperature

        # Load LLM as policy
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Log partition function Z (trainable parameter)
        self.logZ = nn.Parameter(torch.tensor(5.0, requires_grad=True))

    def sample_trajectory(self, prompt: str, max_steps: int = 5) -> Tuple[List[str], List[str], torch.Tensor]:
        """
        Samples a trajectory using the LLM as a forward policy.

        Args:
            prompt: Initial state description.
            max_steps: Maximum number of actions in the trajectory.

        Returns:
            actions: List of sampled actions.
            states: List of generated states.
            log_pf: Log probability of the trajectory.
        """
        tokenized_prompt = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        actions, states = [], []
        log_probs = []

        for step in range(max_steps):
            outputs = self._model.generate(**tokenized_prompt, max_new_tokens=10, return_dict_in_generate=True, output_scores=True)
            generated_tokens = outputs.sequences[:, tokenized_prompt["input_ids"].shape[1]:]
            generated_text = self._tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

            actions.append(generated_text)
            states.append(generated_text)  # Simplified state tracking

            # Compute token-wise log probability
            logits = torch.stack(outputs.scores, dim=1)  # Shape: (batch, seq_len, vocab_size)
            probs = torch.softmax(logits, dim=-1)
            token_log_probs = torch.log(probs[:, torch.arange(probs.shape[1]), generated_tokens])  # Log probability of generated tokens
            log_probs.append(token_log_probs.sum())  # Sum log probs over the action sequence

            # Update prompt for next step
            tokenized_prompt = self._tokenizer(generated_text, return_tensors="pt").to(self.device)

        log_pf = torch.stack(log_probs).sum()  # Total log probability of trajectory
        return actions, states, log_pf

    def compute_reward(self, states: List[str], goal: str) -> torch.Tensor:
        """
        Computes reward based on the final state.

        Args:
            states: List of generated states.
            goal: Goal condition to evaluate.

        Returns:
            Reward tensor.
        """
        final_state = states[-1]
        goal_met = goal in final_state
        reward = 100 if goal_met else 10 * sum(1 for g in goal.split() if g in final_state) / len(goal.split())
        return torch.tensor(reward).to(self.device)

    def trajectory_balance_loss(self, log_pf: torch.Tensor, log_r: torch.Tensor) -> torch.Tensor:
        """
        Computes the trajectory balance loss.

        Args:
            log_pf: Log probability of the forward trajectory.
            log_r: Log probability of the reward function.

        Returns:
            Loss tensor.
        """
        return ((log_pf + self.logZ - log_r) ** 2).mean()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Performs a single training step.

        Args:
            batch: Dictionary containing initial states and goals.

        Returns:
            Loss tensor.
        """
        initial_state = batch["initial_state"]
        goal = batch["goal"]

        actions, states, log_pf = self.sample_trajectory(initial_state)

        # Compute rewards
        reward = self.compute_reward(states, goal)
        log_r = torch.log(reward + 1e-4)

        # Compute loss
        loss = self.trajectory_balance_loss(log_pf, log_r)
        return loss

    def configure_optimizers(self):
        """
        Configures optimizer for training.

        Returns:
            Optimizer instance.
        """
        return torch.optim.AdamW([{"params": self._model.parameters(), "lr": 1e-4}, {"params": [self.logZ], "lr": 1e-1}])

    @property
    def get_model(self) -> PreTrainedModel:
        return self._model

    @property
    def get_tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer
