#!/bin/bash
#SBATCH --job-name=evo_linear
#SBATCH --account=soc-np
#SBATCH --partition=soc-np
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

script_dir="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/EVO/EVO_embed_classify"

cd ${script_dir}

# Setup environment
module load cuda
nvidia-smi
source activate evo

echo "TIME: START SVM training  = `date +"%Y-%m-%d %T"`"

# Create logs directory if it doesn't exist
mkdir -p logs

# Set paths
INPUT_DIR="$script_dir/CSV"  
OUTPUT_DIR="$script_dir/linear_$(date +%Y%m%d_%H%M%S)"
EMBEDDINGS_DIR="$script_dir/embeddings" 

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
