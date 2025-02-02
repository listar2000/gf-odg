# 🌊 GF-ODG: GFlowNet for Open-ended Diverse Generation

> Leveraging GFlowNets for controlled, diverse text generation with language models

## 📚 Table of Contents
- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Setup](#-setup)
- [Configuration](#-configuration)
- [Development](#-development)
- [GFlowNet Experiments](#-gflownet-experiments)
- [Model Downloading](#-model-downloading)
- [Chatbot Demo](#-chatbot-demo)

## 🎯 Overview

GF-ODG is a framework for training language models to generate diverse, controlled text using GFlowNet-based training. The project aims to:

- 🎨 Enable open-ended creative generation
- 🔒 Respect hard constraints in generation
- 🎲 Balance diversity across different generation modes
- 📈 Scale efficiently with large language models

## 🗂 Project Structure

```
gf-odg/
├── configs/                  # Configuration management
│   ├── models/              # Model-specific configs
│   │   ├── llama3-8b.yaml   # Llama 3 8B settings
│   │   └── phi-4.yaml       # Phi 4 settings
│   └── main.yaml            # Global configuration
│
├── models/                  # Model management
│   ├── .cache/             # HuggingFace model cache
│   ├── base_models/        # Base pretrained models
│   ├── adapted_models/     # PEFT-adapted models
│   └── checkpoints/        # Training checkpoints
│
├── scripts/                # Utility scripts
│   └── download_models.py  # Model download script
│
└── src/                    # Source code
    └── demos/              # Demo applications
        └── chatbot_demo.py # Chatbot demo script
```

## 🚀 Setup

1. **Environment Setup**

```bash
# Create and activate UV environment
uv venv
source .venv/bin/activate.fish

# Install dependencies
uv pip install -r uv.yaml
```

2. **Environment Variables**

Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```

Required credentials:
- `HF_TOKEN`: HuggingFace access token
- `WANDB_API_KEY`: Weights & Biases API key
- `WANDB_ENTITY`: Your W&B username

## ⚙️ Configuration

The project uses a hierarchical configuration system powered by Hydra:

1. **Global Settings** (`configs/main.yaml`):
   ```yaml
   paths:
     root: ${oc.env:PROJECT_ROOT}
     models: ${paths.root}/models
     hf_cache: ${paths.models}/.cache
   ```

2. **Model Configurations** (`configs/models/*.yaml`):
   ```yaml
   model:
     name: "meta-llama/Meta-Llama-3-8B-Instruct"
     type: "causal"
     trust_remote_code: true
   ```

## 🛠 Development

### Model Loading

```python
from transformers import AutoModelForCausalLM
import hydra

@hydra.main(config_path="configs", config_name="main")
def main(cfg):
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        **cfg.model.model_kwargs
    )
```

### Training Configuration

Override configs via command line:
```bash
python train.py model=llama3-8b
```

## 📝 GFlowNet Experiments

This repository contains experiments and demos using GFlowNet and large language models.

### Setup

1. Clone the repository
2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install uv
   uv pip install -r uv.yaml
   ```

### Model Downloading

The repository includes a convenient script to download models from Hugging Face. To use it:

1. Set up your Hugging Face token:
   ```bash
   export HF_TOKEN=your_token_here
   ```

2. Run the download script:
   ```bash
   python scripts/download_models.py
   ```

3. Follow the interactive prompts to select which model to download. Available models include:
   - Llama 3 8B (Instruct) - Recommended!
   - Phi 4

The models will be downloaded to the `models/pretrained` directory.

### Chatbot Demo

A terminal-based chatbot demo is available using the downloaded models. The demo features:
- Beautiful CLI interface with Rich
- Chat history support
- Markdown formatting
- Emoji support

To run the chatbot:

1. Make sure you have downloaded a model first (see above)
2. Run the demo script:
   ```bash
   python src/demos/chatbot_demo.py
   ```

#### Chatbot Commands
- Type your message and press Enter to chat
- Type `exit` to end the chat session
- Type `reset` to clear the conversation history

## 📝 Notes

- Keep large files in `models/` directory (git-ignored)
- Use Hydra for configuration management
- Follow the project structure for new additions