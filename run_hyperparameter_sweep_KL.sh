#!/bin/bash

# Base model path - update this to your model path
MODEL_PATH="/net/scratch/llama3/Meta-Llama-3-8B-Instruct"

# Base prompt
PROMPT="Generate 5 random numbers from 1 to 5, independently of each other, separated by commas. The generated numbers:"

# Base output directory
BASE_OUTPUT_DIR="/home/jiaweizhang/gf-odg/models/finetuned/train_number"

mkdir -p ${BASE_OUTPUT_DIR}

# Slurm configuration
SLURM_CPUS=16
SLURM_MEM=64000
SLURM_GPU="a100:1"

# Export WANDB API key

# Create the output directory and logs if not present
mkdir -p ${BASE_OUTPUT_DIR}/logs

# Define output directory and run name for this single job
output_dir="${BASE_OUTPUT_DIR}/number"
run_name="number"

# Array of KL_penalty values to test
KL_PENALTY_VALUES=(0.0001 0.001 0.005 0.01 0.02 0.04 0.08)

# Loop through each KL_penalty value and submit a job
for kl_penalty in "${KL_PENALTY_VALUES[@]}"; do
    # Create a unique output directory and run name based on KL_penalty
    output_dir="${BASE_OUTPUT_DIR}/kl_${kl_penalty}"
    run_name="number_kl_${kl_penalty}"

    # Create the job script
    job_script=$(mktemp)

    cat > "$job_script" << EOL
#!/bin/bash
#SBATCH --job-name=${run_name}
#SBATCH --output=${BASE_OUTPUT_DIR}/logs/${run_name}_%j.out
#SBATCH --error=${BASE_OUTPUT_DIR}/logs/${run_name}_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --gres=gpu:a100:1
#SBATCH --time=120:00
#SBATCH --partition=general

# Base model path - update this to your model path
MODEL_PATH="/net/scratch/llama3/Meta-Llama-3-8B-Instruct"

# Base prompt
PROMPT="Generate 5 random numbers from 1 to 5, independently of each other, separated by commas. The generated numbers:"

# Base output directory
BASE_OUTPUT_DIR="/home/jiaweizhang/gf-odg/models/finetuned/train_number"

# Create output and log directories if they don't exist
mkdir -p ${output_dir}
mkdir -p ${BASE_OUTPUT_DIR}/logs

export WANDB_API_KEY="94df40f69fe1711f227d8df8c9cf9ea389060b66"

# Activate your environment if needed
eval "\$(~/miniconda3/bin/conda shell.bash hook)"  # Adjust path if needed
conda activate FoR

# Run the training script with the necessary hyperparameters
python /home/jiaweizhang/gf-odg/src/gflownet/train_number.py \
    --model_name_or_path ${MODEL_PATH} \
    --prompt "${PROMPT}" \
    --concept_name "animal" \
    --w_kl ${kl_penalty} \
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

EOL

    # Submit the job
    echo "Submitting the job for ${run_name}"
    sbatch "$job_script"

    # Clean up the temporary job script
    rm "$job_script"

    echo "Job submitted for kl_penalty=${kl_penalty}!"
done
