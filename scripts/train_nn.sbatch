#!/bin/bash
#SBATCH --job-name=evo_nn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/nn_%j.out
#SBATCH --error=logs/nn_%j.err

# Change to project root directory (parent of scripts/)
cd ${SLURM_SUBMIT_DIR}/..

# Load required modules
module load python/3.9
module load cuda/11.7  # Adjust CUDA version as needed

# Activate conda environment if needed
# source activate your_env_name

# Create logs directory if it doesn't exist
mkdir -p logs

# Get seed from command line argument
SEED=$1

# Set paths
INPUT_DIR="./data"  # Update this path
OUTPUT_DIR="./results/nn_seed${SEED}"  # Results will be saved here with seed in path
EMBEDDINGS_DIR="./embeddings"  # Shared embeddings directory

# Create directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${EMBEDDINGS_DIR}

# Check GPU availability
echo "GPU Information:"
nvidia-smi
echo "----------------------------------------"

# Run training for Neural Network classifier only
echo "Starting Neural Network Classifier training at $(date)"
echo "Random seed: ${SEED}"
echo "Input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Embeddings directory: ${EMBEDDINGS_DIR}"
echo "----------------------------------------"

python train_evo.py \
    --input_dir ${INPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --embeddings_dir ${EMBEDDINGS_DIR} \
    --classifiers nn \
    --device cuda \
    --pooling mean \
    --seed ${SEED}

echo "----------------------------------------"
echo "Neural Network Classifier training completed at $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
