#!/usr/bin/env python3
"""
Interactive chatbot demo using our downloaded LLM.
Uses Hydra for configuration and Rich for a beautiful CLI interface.
"""

import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from peft import PeftModel
from typing import List, Tuple
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.markdown import Markdown


# Disable tokenizer warnings
logging.set_verbosity_error()


class Chatbot:
    def __init__(self, cfg: DictConfig):
        """Initialize chatbot with configuration."""
        self.model_cfg = cfg.demos.model
        self.chat_cfg = cfg.demos.chat  # Get chat config
        self.console = Console()
        self.history: List[Tuple[str, str]] = []

        # Ask user about model type
        use_finetuned = Confirm.ask(
            "Would you like to use a fine-tuned model?")

        # Load model and tokenizer
        with self.console.status("[bold green]Loading model and tokenizer...", spinner="dots"):
            self.console.print(f"📂 Loading from: {
                               self.model_cfg.model.pretrained_path}")

            # Load base model with eager attention
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_cfg.model.pretrained_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_cfg.model.pretrained_path,
                local_files_only=True,
                padding_side=self.model_cfg.model.padding_side,
                # pad_token_id=self.model_cfg.model.pad_token_id
            )

            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Load LoRA weights if requested
            if use_finetuned:
                adapter_path = Prompt.ask(
                    "[bold yellow]Please enter the path to the LoRA adapter:",
                    default="/net/scratch2/listar2000/gfn-od/models/finetuned/train_animal/w_o_0.9"
                )

                self.console.print(
                    f"[bold green]Loading LoRA weights from: {adapter_path}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    adapter_path
                )

        self.console.print("[bold green]✨ Model loaded successfully!")

    def format_prompt(self, user_input: str) -> str:
        """Format the chat prompt with history."""
        # Format history
        history_str = ""
        if self.history:
            for human, assistant in self.history:
                history_str += f"\nHuman: {human}\nAssistant: {assistant}"

        # Return formatted prompt using system template from config
        return self.chat_cfg.system_template.format(
            history=history_str,
            input=user_input
        )

    def display_message(self, role: str, content: str, color: str = "white") -> None:
        """Display a chat message in a nice panel."""
        emoji = "🧑" if role == "Human" else "🤖"
        self.console.print(Panel(
            Markdown(content),
            title=f"{emoji} {role}",
            border_style=color
        ))

    def run(self):
        self.console.print("[bold green]✨ Chatbot is ready!")
        self.console.print(
            "[italic]Type 'quit' to exit or 'clear' to clear history[/italic]\n")

        while True:
            # Get user input
            user_input = Prompt.ask("[bold yellow]You")
            if user_input.lower() == 'quit':
                self.console.print("[bold green]👋 Goodbye!")
                break
            elif user_input.lower() == 'clear':
                self.history = []
                self.console.print("[bold green]🔄 Chat history cleared!")
                continue

            # Display user message
            self.display_message("Human", user_input, "yellow")

            # Format prompt and generate response
            # prompt = self.format_prompt(user_input)
            prompt = user_input
            inputs = self.tokenizer(
                prompt, return_tensors="pt").to(self.model.device)

            # Generate with status spinner
            with self.console.status("[bold blue]🤔 Thinking...", spinner="dots"):
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=40,
                        temperature=1.0,
                        top_p=0.95,
                        repetition_penalty=1.1,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        stop_strings=["\n", ".\n\n", ".\n"],
                        tokenizer=self.tokenizer
                    )

                # Decode and format response
                response = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True)
                answer = response[len(prompt):].strip()

            # Display assistant message
            self.display_message("Assistant", answer, "green")

            # Update history
            self.history.append((user_input, answer))
            if len(self.history) > 10:  # Keep last 10 turns
                self.history.pop(0)


@hydra.main(config_path="../../configs", config_name="demos/chatbot", version_base="1.1")
def main(config: DictConfig):
    try:
        chatbot = Chatbot(config)
        chatbot.run()
    except KeyboardInterrupt:
        Console().print("\n[bold green]👋 Chat ended by user. Goodbye!")
    except Exception as e:
        Console().print(f"\n[bold red]❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
