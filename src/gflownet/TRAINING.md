# Training Script Documentation

## Overview
This document provides information about the training script (`train_animal.py`) and its features.

## Code Structure
The training script has been refactored to use a more organized parameter structure with dataclasses:

1. **ModelConfig**: Configuration for model and tokenizer
   ```python
   @dataclass
   class ModelConfig:
       model: PeftModel
       tokenizer: AutoTokenizer
       text_processor: RawTextProcessor
       sentence_transformer: Optional[SentenceTransformer] = None
   ```

2. **TextGenerationConfig**: Configuration for text generation
   ```python
   @dataclass
   class TextGenerationConfig:
       prompt: str
       max_new_tokens: int = 40
       batch_size: int = 16
       generation_config: Optional[GenerationConfig] = None
   ```

3. **DiversityConfig**: Configuration for diversity training
   ```python
   @dataclass
   class DiversityConfig:
       concept_name: str
       n_clusters: int = 5
       w_c: float = 0.5  # Weight for concept loss
       w_o: float = 0.5  # Weight for open block loss
       num_samples: int = 100  # Number of samples for initializing replay buffer
       buffer_size: int = 500  # Maximum size of the replay buffer
       update_clusters_every: int = 20  # Update clusters every N samples
       min_samples_for_clustering: int = 10  # Minimum samples required for clustering
   ```

4. **TrainingConfig**: Configuration for training process
   ```python
   @dataclass
   class TrainingConfig:
       num_epochs: int = 10
       num_steps_per_epoch: int = 10
       learning_rate: float = 5e-5
       final_learning_rate: float = 1e-6
       warmup_steps: int = 0
       lr_scheduler_type: str = "cosine"
       output_dir: str = "diversity_model"
   ```

5. **WandbConfig**: Configuration for Weights & Biases logging
   ```python
   @dataclass
   class WandbConfig:
       use_wandb: bool = False
       wandb_project: str = "gfn-diversity"
       wandb_name: Optional[str] = None
   ```

## Features

### Weights & Biases Integration
The training script supports logging metrics and samples to Weights & Biases (wandb). This allows for better tracking of training progress and results.

To enable wandb logging, set the `use_wandb` parameter to `True` in the `WandbConfig`. You can also specify the project name and run name.

Example:
```python
wandb_config = WandbConfig(
    use_wandb=True,
    wandb_project="gfn-diversity",
    wandb_name="train_animal"
)
```

The following metrics are logged to wandb:
- Total loss
- Concept loss
- Open block loss
- Learning rate
- Sample generations

### Learning Rate Scheduler
The training script supports various learning rate schedulers to improve training stability and performance. The following schedulers are available:

1. **Cosine Scheduler**: Gradually decreases the learning rate following a cosine curve.
2. **Linear Scheduler**: Linearly decreases the learning rate from the initial value to the final value.
3. **Constant Scheduler**: Maintains a constant learning rate throughout training.

Both the cosine and linear schedulers support warmup steps, where the learning rate gradually increases from 0 to the initial learning rate over a specified number of steps.

Example:
```python
training_config = TrainingConfig(
    learning_rate=1e-4,
    final_learning_rate=1e-6,
    warmup_steps=100,
    lr_scheduler_type="cosine"
)
```

### Replay Buffer Configuration
The replay buffer is used to store and manage generated samples during training. It now has configurable parameters:

- `buffer_size`: Maximum number of samples to store in the buffer
- `update_clusters_every`: How often to update the clusters (in number of samples)
- `min_samples_for_clustering`: Minimum number of samples required before clustering

Example:
```python
diversity_config = DiversityConfig(
    concept_name="animal",
    n_clusters=5,
    num_samples=160,
    w_c=0.8,
    w_o=0.2,
    buffer_size=500,
    update_clusters_every=20,
    min_samples_for_clustering=10
)
```

## Command-Line Arguments

The training script now supports command-line arguments to override default configuration values. This makes it easy to run experiments with different hyperparameters without modifying the code.

### Usage

```bash
python train_animal.py [OPTIONS]
```

### Available Options

#### Model and Tokenizer Arguments
- `--model_name_or_path`: Model name or path (default: "gpt2")
- `--lora_r`: LoRA r dimension (default: 16)
- `--lora_alpha`: LoRA alpha parameter (default: 32)
- `--lora_dropout`: LoRA dropout rate (default: 0.05)

#### Text Generation Arguments
- `--prompt`: Prompt for text generation (default: "The animal")
- `--batch_size`: Batch size for generation (default: 32)
- `--max_new_tokens`: Maximum number of new tokens to generate (default: 30)

#### Diversity Config Arguments
- `--concept_name`: Name of the concept to diversify (default: "animal")
- `--n_clusters`: Number of clusters for diversity (default: 5)
- `--num_samples`: Number of samples for initializing replay buffer (default: 320)
- `--w_c`: Weight for concept loss (default: 0.8)
- `--w_o`: Weight for open block loss (default: 0.2)
- `--buffer_size`: Maximum size of the replay buffer (default: 500)
- `--update_clusters_every`: Update clusters every N samples (default: 100)
- `--min_samples_for_clustering`: Minimum samples required for clustering (default: 20)

#### Training Config Arguments
- `--num_epochs`: Number of training epochs (default: 10)
- `--num_steps_per_epoch`: Number of steps per epoch (default: 10)
- `--learning_rate`: Initial learning rate (default: 1e-4)
- `--final_learning_rate`: Final learning rate (default: 3e-5)
- `--warmup_steps`: Number of warmup steps (default: 0)
- `--lr_scheduler_type`: Type of learning rate scheduler (choices: "cosine", "linear", "constant", default: "cosine")
- `--output_dir`: Directory to save the model (default: "diversity_model")

#### Wandb Config Arguments
- `--use_wandb`: Whether to use Weights & Biases for logging (flag, no value needed)
- `--wandb_project`: Weights & Biases project name (default: "gfn-diversity")
- `--wandb_name`: Weights & Biases run name (default: "train_animal")

### Example

```bash
python train_animal.py \
    --model_name_or_path "/path/to/model" \
    --prompt "In 20 words, say whether you love dog or cat more, and give a reason; Answer:" \
    --concept_name "animal" \
    --n_clusters 5 \
    --num_samples 320 \
    --w_c 0.7 \
    --w_o 0.3 \
    --buffer_size 500 \
    --update_clusters_every 100 \
    --min_samples_for_clustering 20 \
    --batch_size 32 \
    --max_new_tokens 30 \
    --num_epochs 10 \
    --num_steps_per_epoch 10 \
    --learning_rate 1e-4 \
    --final_learning_rate 3e-5 \
    --output_dir "output/train_animal_w_o_0.3" \
    --use_wandb \
    --wandb_project "gfn-diversity" \
    --wandb_name "train_animal_w_o_0.3"
```

## Hyperparameter Tuning

A shell script (`run_hyperparameter_sweep.sh`) is provided to automatically submit multiple Slurm jobs with different hyperparameter values. This is particularly useful for tuning the `w_c` and `w_o` parameters, which control the balance between concept loss and open block loss.

### Usage

```bash
./run_hyperparameter_sweep.sh
```

### Configuration

The script is configured to run experiments with `w_o` values ranging from 0.2 to 0.9 (in increments of 0.1), with corresponding `w_c` values such that `w_c + w_o = 1.0`.

Each experiment is submitted as a separate Slurm job with the following resources:
- 32 CPUs per task
- 64GB of memory
- 1 A100 GPU

### Customization

You can modify the script to change:
- The range of `w_o` values to test
- The base model path
- The prompt
- The output directory
- The Slurm resource requirements
- Any other hyperparameters

## Installation
Make sure to install the required dependencies:
```bash
pip install -r requirements.txt
