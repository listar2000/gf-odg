#!/bin/bash

# Base model path - update this to your model path
MODEL_PATH="/net/scratch/llama3/Meta-Llama-3-8B-Instruct"

# Base prompt
PROMPT="Generate 9 random numbers from 1-9 independently of each other, ensuring no influence between selections. Separate the numbers with commas. Answer:"

# Base output directory
BASE_OUTPUT_DIR="/home/jiaweizhang/gf-odg/models/finetuned/train_animal"

mkdir -p ${BASE_OUTPUT_DIR}

# Slurm configuration (not necessary for direct run, but you can keep them if you want to track)
SLURM_CPUS=16
SLURM_MEM=64000
SLURM_GPU="a100:1"

# Export WANDB API key
export WANDB_API_KEY=94df40f69fe1711f227d8df8c9cf9ea389060b66

# Create the output directory and logs if not present
mkdir -p ${BASE_OUTPUT_DIR}/logs

# Define output directory and run name for this single job
output_dir="${BASE_OUTPUT_DIR}/flower"
run_name="flower"

# Create output and log directories if they don't exist
mkdir -p ${output_dir}
mkdir -p ${BASE_OUTPUT_DIR}/logs

# Run the training script with the necessary hyperparameters
python /home/jiaweizhang/gf-odg/src/gflownet/train_flower.py \
    --model_name_or_path ${MODEL_PATH} \
    --prompt "${PROMPT}" \
    --concept_name "animal" \
    --n_clusters 5 \
    --num_samples 320 \
    --buffer_size 500 \
    --update_clusters_every 100 \
    --min_samples_for_clustering 20 \
    --batch_size 32 \
    --max_new_tokens 30 \
    --num_epochs 10 \
    --num_steps_per_epoch 10 \
    --learning_rate 1e-4 \
    --final_learning_rate 3e-5 \
    --warmup_steps 0 \
    --lr_scheduler_type "cosine" \
    --output_dir ${output_dir} \
    --use_wandb \
    --wandb_project "gfn-diversity" \
    --wandb_name ${run_name}

echo "Job completed!"
