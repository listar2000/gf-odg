# 🌊 GF-ODG: GFlowNet for Open-ended Diverse Generation

> Leveraging GFlowNets for controlled, diverse text generation with language models

## 📚 Table of Contents
- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Setup](#-setup)
- [Configuration](#-configuration)
- [Development](#-development)

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
│   │   ├── gemma-2b.yaml    # Gemma 2B settings
│   │   └── llama3-8b.yaml   # LLaMA 3 settings
│   └── main.yaml            # Global configuration
│
├── models/                  # Model management
│   ├── .cache/             # HuggingFace model cache
│   ├── base_models/        # Base pretrained models
│   ├── adapted_models/     # PEFT-adapted models
│   └── checkpoints/        # Training checkpoints
│
└── scripts/                # Utility scripts
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
     name: "google/gemma-2b"
     type: "causal"
     model_kwargs:
       load_in_4bit: true
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
python train.py model=gemma-2b
```

## 📝 Notes

- Keep large files in `models/` directory (git-ignored)
<!-- - Use 4-bit quantization for memory efficiency
- Store experiment results in `outputs/` (git-ignored) -->

<!-- ## 🔗 References

$\text{GFlowNet}$ formulation:

$$
P_F(s_{t+1}|s_t) = \frac{F(s_t \rightarrow s_{t+1})}{F(s_t)} = \frac{R(s_{t+1})^{1/\beta}}{R(s_t)^{1/\beta}}
$$

Where:
- $P_F$: Forward policy
- $s_t$: State at time $t$
- $R$: Reward function
- $\beta$: Temperature parameter -->