#!/bin/bash
#SBATCH --job-name=evo_linear
#SBATCH --partition=norm
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/linear_%j.out
#SBATCH --error=logs/linear_%j.err

# Change to project root directory (parent of scripts/)
cd ${SLURM_SUBMIT_DIR}/..

# Load required modules
module load python/3.9

# Activate conda environment if needed
# source activate your_env_name

# Create logs directory if it doesn't exist
mkdir -p logs

# Set paths
INPUT_DIR="./data"  # Update this path
OUTPUT_DIR="./results/linear"  # Results will be saved here
EMBEDDINGS_DIR="./embeddings"  # Shared embeddings directory

# Create directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${EMBEDDINGS_DIR}

# Run training for Linear classifier only
echo "Starting Linear Classifier training at $(date)"
echo "Input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Embeddings directory: ${EMBEDDINGS_DIR}"
echo "----------------------------------------"

python train_evo.py \
    --input_dir ${INPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --embeddings_dir ${EMBEDDINGS_DIR} \
    --classifiers linear \
    --device cpu \
    --pooling mean

echo "----------------------------------------"
echo "Linear Classifier training completed at $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
