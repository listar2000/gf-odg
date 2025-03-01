#!/bin/bash

# Base model path - update this to your model path
MODEL_PATH="/net/scratch2/listar2000/gfn-od/models/pretrained/Meta-Llama-3-8B-Instruct"

# Base prompt
PROMPT="Generate a synthetic profile of a US politician. Your response must follow this exact structured format: {Party}{Home State}{Political Agenda} Where: Party: choose exactly one from Democrat or Republican. Home State: Choose exactly one from California, Texas, Michigan, Illinois. Political Agenda: Write a concise, one-sentence description of the politician’s main political agenda. Some examples: {Republican}{Alabama}{Less taxation for business}, {Democrat}{Washington}{Climate change}; Answer {"

# Base output directory
BASE_OUTPUT_DIR="/net/scratch2/listar2000/gfn-od/models/finetuned/train_biography"

# Slurm configuration
SLURM_CPUS=32
SLURM_MEM=64000
SLURM_GPU="a100:1"

# Array of w_o values to test
W_O_VALUES=(0.3 0.4 0.5 0.6 0.7 0.8)

# Loop through each w_o value and submit a job
for w_o in "${W_O_VALUES[@]}"; do
    # Calculate w_c (they sum to 1)
    w_c=$(echo "1.0 - $w_o" | bc)
    
    # Create a unique output directory and run name based on the w_o value
    output_dir="${BASE_OUTPUT_DIR}/bio_w_o_${w_o}"
    run_name="bio_w_o_${w_o}"
    
    # Create the job script
    job_script=$(mktemp)
    
    cat > "$job_script" << EOL
#!/bin/bash
#SBATCH --job-name=${run_name}
#SBATCH --output=${BASE_OUTPUT_DIR}/logs/${run_name}_%j.out
#SBATCH --error=${BASE_OUTPUT_DIR}/logs/${run_name}_%j.err
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
python /net/scratch2/listar2000/gfn-od/src/gflownet/train_biography.py \
    --model_name_or_path ${MODEL_PATH} \
    --prompt "${PROMPT}" \
    --concept_name "biography" \
    --num_samples 320 \
    --w_c ${w_c} \
    --w_o ${w_o} \
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
