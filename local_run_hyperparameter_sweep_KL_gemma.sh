#!/bin/bash

# Base model path - update this to your model path
MODEL_PATH="/net/scratch/jiaweizhang/gemma2"

# Base prompt
PROMPT="Generate 6 random numbers from 1 to 6, independently of each other, separated by commas. The generated numbers:"

# Base output directory
BASE_OUTPUT_DIR="/home/jiaweizhang/gf-odg/models/finetuned/train_number_6_lr3e-5_gemma2"

mkdir -p ${BASE_OUTPUT_DIR}

# Create the output directory and logs if not present
mkdir -p ${BASE_OUTPUT_DIR}/logs

# Define output directory and run name for this single job
output_dir="${BASE_OUTPUT_DIR}/number"
run_name="number"

# Array of KL_penalty values to test
KL_PENALTY_VALUE=0.0000001

source /net/scratch/jiaweizhang/llm_finetune_env/bin/activate

python /home/jiaweizhang/gf-odg/src/gflownet/train_number.py \
    --model_name_or_path ${MODEL_PATH} \
    --prompt "${PROMPT}" \
    --w_kl ${KL_PENALTY_VALUE} \
    --batch_size 32 \
    --max_new_tokens 20 \
    --num_epochs 10 \
    --num_steps_per_epoch 10 \
    --learning_rate 3e-5 \
    --final_learning_rate 3e-6 \
    --warmup_steps 0 \
    --lr_scheduler_type "cosine" \
    --output_dir ${output_dir} \
    --use_wandb \
    --wandb_project "gfn-diversity" \
    --wandb_name ${run_name}

