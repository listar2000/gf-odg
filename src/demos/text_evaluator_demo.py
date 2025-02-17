#!/usr/bin/env python3
"""
Text evaluation demo using LLM to assess creativity and openness of texts.
Uses Hydra for configuration and Rich for a beautiful CLI interface.
"""

import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from typing import Dict, List, Tuple

# Disable tokenizer warnings
logging.set_verbosity_error()

class TextEvaluator:
    def __init__(self, cfg: DictConfig):
        """Initialize text evaluator with configuration."""
        self.model_cfg = cfg.model
        self.eval_cfg = cfg.evaluation
        self.console = Console()
        
        # Load model and tokenizer
        with self.console.status("[bold green]Loading model and tokenizer...", spinner="dots"):
            self.console.print(f"📂 Loading from: {self.model_cfg.model.pretrained_path}")
            
            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_cfg.model.pretrained_path,
                device_map="auto",
                local_files_only=True,
                trust_remote_code=self.model_cfg.model.trust_remote_code
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_cfg.model.pretrained_path,
                local_files_only=True,
                padding_side=self.model_cfg.model.padding_side
            )
            
            if not self.tokenizer.pad_token_id:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _construct_prompt(self, text: str) -> str:
        """Construct the evaluation prompt."""
        criteria_str = "\n".join([f"- {k}: {v}" for k, v in self.eval_cfg.criteria.items()])
        prompt = f"""{self.eval_cfg.system_prompt}

Please evaluate the following text based on these criteria:
{criteria_str}

For each criterion, provide:
1. A score (1-10)
2. A brief explanation

Text to evaluate:
{text}

Evaluation:"""
        return prompt

    def _generate_response(self, prompt: str) -> str:
        """Generate evaluation response from the model."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.eval_cfg.generation
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()

    def _display_evaluation(self, evaluation: str):
        """Display the evaluation results in a beautiful format."""
        panel = Panel(
            Markdown(evaluation),
            title="[bold blue]📝 Evaluation Results",
            border_style="blue",
            padding=(1, 2)
        )
        self.console.print(panel)

    def run(self):
        """Run the text evaluation demo."""
        self.console.print("[bold blue]🤖 Welcome to the Text Evaluator![/bold blue]")
        self.console.print("I'll help you evaluate the creativity and openness of your text.")
        
        while True:
            # Get text input
            text = Prompt.ask("\n[bold green]Please enter the text to evaluate[/bold green]"
                            "\n(or type 'exit' to quit)")
            
            if text.lower() == 'exit':
                break
            
            # Generate and display evaluation
            with self.console.status("[bold green]Analyzing text...", spinner="dots"):
                prompt = self._construct_prompt(text)
                evaluation = self._generate_response(prompt)
                self._display_evaluation(evaluation)
            
            self.console.print("\n[italic]Press Enter to continue or type 'exit' to quit[/italic]")

@hydra.main(config_path="../../configs", config_name="main", version_base="1.1")
def main(config: DictConfig):
    try:
        evaluator = TextEvaluator(config.text_evaluator)
        evaluator.run()
    except KeyboardInterrupt:
        Console().print("\n[bold green]👋 Evaluation ended by user. Goodbye!")
    except Exception as e:
        Console().print(f"\n[bold red]❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
