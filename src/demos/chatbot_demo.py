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
from typing import List, Tuple
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
import time

# Disable tokenizer warnings
logging.set_verbosity_error()

# Add project root to path so we can import our modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

class Chatbot:
    def __init__(self, cfg: DictConfig):
        """Initialize chatbot with configuration."""
        self.model_cfg = cfg.demos.model
        self.chat_cfg = cfg.demos.chat
        self.history: List[Tuple[str, str]] = []
        self.console = Console()
        
        # Load model and tokenizer
        with self.console.status("[bold green]Loading model and tokenizer...", spinner="dots"):
            self.console.print(f"📂 Loading from: {self.model_cfg.model.pretrained_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_cfg.model.pretrained_path,
                local_files_only=True,
                padding_side=self.model_cfg.model.padding_side,
                pad_token_id=self.model_cfg.model.pad_token_id
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_cfg.model.pretrained_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True
            )
            
            # Ensure padding token is set
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.model_cfg.model.pad_token_id
                
        self.console.print("[bold green]✨ Model loaded successfully!")
    
    def format_prompt(self, user_input: str) -> str:
        """Format the chat prompt with history."""
        # Format history
        history_str = ""
        if self.history:
            for human, assistant in self.history:
                history_str += f"\nHuman: {human}\nAssistant: {assistant}"
        
        # Return formatted prompt
        return self.chat_cfg.system_template.format(
            history=history_str,
            input=user_input
        )
    
    def generate_response(self, user_input: str) -> str:
        """Generate response for user input."""
        if not user_input.strip():
            return "Please enter a message! 📝"
            
        # Format prompt with history
        prompt = self.format_prompt(user_input)
        
        # Tokenize and generate with a nice spinner
        with self.console.status("[bold yellow]🤔 Thinking...", spinner="dots"):
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.chat_cfg.generation
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            assistant_response = full_response[len(prompt):].strip()
        
        # Update history
        self.history.append((user_input, assistant_response))
        if len(self.history) > self.chat_cfg.history.max_turns:
            self.history.pop(0)
        
        return assistant_response
    
    def display_message(self, role: str, content: str, color: str = "white") -> None:
        """Display a chat message in a nice panel."""
        emoji = "🧑" if role == "Human" else "🤖"
        self.console.print(Panel(
            Markdown(content),
            title=f"{emoji} {role}",
            border_style=color
        ))
    
    def run_chat_loop(self) -> None:
        """Run the main chat loop."""
        # Show welcome message
        self.console.print("\n[bold cyan]Welcome to the GFlowNet Chatbot! 🌟[/bold cyan]")
        self.console.print("[italic]Type 'exit' to end the chat or 'reset' to clear history[/italic]\n")
        
        while True:
            # Get user input
            user_input = Prompt.ask("\n[bold blue]You").strip()
            
            # Handle special commands
            if user_input.lower() == "exit":
                self.console.print("\n[bold cyan]👋 Goodbye! Have a great day![/bold cyan]")
                break
            elif user_input.lower() == "reset":
                self.history = []
                self.console.print("\n[bold green]🔄 Chat history has been reset![/bold green]")
                continue
            
            # Display user message
            self.display_message("Human", user_input, "blue")
            
            # Generate and display response
            response = self.generate_response(user_input)
            self.display_message("Assistant", response, "green")

@hydra.main(config_path="../../configs", config_name="demos/chatbot")
def main(cfg: DictConfig) -> None:
    """Run the chatbot demo."""
    try:
        # Initialize and run chatbot
        chatbot = Chatbot(cfg)
        chatbot.run_chat_loop()
    except KeyboardInterrupt:
        console = Console()
        console.print("\n[bold cyan]👋 Chat ended by user. Goodbye![/bold cyan]")
    except Exception as e:
        console = Console()
        console.print(f"\n[bold red]❌ Error: {str(e)}[/bold red]")
        raise

if __name__ == "__main__":
    main()
