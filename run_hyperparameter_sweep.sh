#!/bin/bash

# Base model path - update this to your model path
MODEL_PATH="/net/scratch2/listar2000/gfn-od/models/pretrained/Meta-Llama-3-8B-Instruct"

# Base prompt
PROMPT="In 20 words, say whether you love dog or cat more, and give a reason; Answer:"

# Base output directory
BASE_OUTPUT_DIR="/net/scratch2/listar2000/gfn-od/models/finetuned/train_animal"

# Slurm configuration
SLURM_CPUS=16
SLURM_MEM=64000
SLURM_GPU="a100:1"

# Array of w_o values to test
W_O_VALUES=(0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Loop through each w_o value and submit a job
for w_o in "${W_O_VALUES[@]}"; do
    # Calculate w_c (they sum to 1)
    w_c=$(echo "1.0 - $w_o" | bc)
    
    # Create a unique output directory and run name based on the w_o value
    output_dir="${BASE_OUTPUT_DIR}/w_o_${w_o}"
    run_name="train_animal_w_o_${w_o}"
    
    # Create the job script
    job_script=$(mktemp)
    
    cat > "$job_script" << EOL
#!/bin/bash
#SBATCH --job-name=train_w_o_${w_o}
#SBATCH --output=${BASE_OUTPUT_DIR}/logs/train_w_o_${w_o}_%j.out
#SBATCH --error=${BASE_OUTPUT_DIR}/logs/train_w_o_${w_o}_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --gres=gpu:${SLURM_GPU}
#SBATCH --time=120:00
#SBATCH --partition=general

# Create output and log directories if they don't exist
mkdir -p ${output_dir}
mkdir -p ${BASE_OUTPUT_DIR}/logs

# Activate your environment if needed
source /net/scratch2/listar2000/gfn-od/.venv/bin/activate

# Run the training script with the specific hyperparameters
python /net/scratch2/listar2000/gfn-od/src/gflownet/train_animal.py \
    --model_name_or_path ${MODEL_PATH} \
    --prompt "${PROMPT}" \
    --concept_name "animal" \
    --n_clusters 5 \
    --num_samples 320 \
    --w_c ${w_c} \
    --w_o ${w_o} \
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
    echo "Submitting job for w_o=${w_o}, w_c=${w_c}"
    sbatch "$job_script"
    
    # Clean up the temporary job script
    rm "$job_script"
    
    # Wait a bit between submissions to avoid overwhelming the scheduler
    sleep 1
done

echo "All jobs submitted!"
