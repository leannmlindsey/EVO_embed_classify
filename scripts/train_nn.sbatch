#!/bin/bash
#SBATCH --job-name=evo_nn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/nn_%j.out
#SBATCH --error=logs/nn_%j.err

# Get seed from command line argument
SEED=$1

script_dir="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/EVO/EVO_embed_classify"

cd ${script_dir}

# Setup environment
module load cuda
nvidia-smi
source activate evo

echo "TIME: START nn training  = `date +"%Y-%m-%d %T"`"

# Create logs directory if it doesn't exist
mkdir -p logs

# Set paths
INPUT_DIR="$script_dir/CSV"  
OUTPUT_DIR="$script_dir/nn_seed{SEED}_$(date +%Y%m%d_%H%M%S)"
EMBEDDINGS_DIR="$script_dir/embeddings" 

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
echo "TIME: END nn training  = `date +"%Y-%m-%d %T"`"
