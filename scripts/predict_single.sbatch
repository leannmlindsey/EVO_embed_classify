#!/bin/bash
#SBATCH --job-name=evo_predict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/predict_%j.out
#SBATCH --error=logs/predict_%j.err

# This sbatch script processes a single CSV file
# Usage: sbatch predict_single.sbatch <input_csv_file>

# Change to project root directory (parent of scripts/)
cd ${SLURM_SUBMIT_DIR}/..

# Load modules
module load python/3.9
module load cuda/11.7

# Activate conda environment
source activate evo

# Get input filename from command line argument
INPUT_CSV=$1

# Check if input file was provided
if [ -z "$INPUT_CSV" ]; then
    echo "Error: No input file provided"
    echo "Usage: sbatch predict_single.sbatch <input_csv_file>"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_CSV" ]; then
    echo "Error: Input file not found: $INPUT_CSV"
    exit 1
fi

# Extract basename without extension for output naming
BASENAME=$(basename "$INPUT_CSV" .csv)

# Set paths
OUTPUT_DIR="./predictions"
MODEL_PATH="./results/nn_seed1/evo_nn_classifier.pt"
SCALER_PATH="./results/nn_seed1/evo_scaler.joblib"
OUTPUT_CSV="${OUTPUT_DIR}/${BASENAME}_predictions.csv"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Print job info
echo "=================================================="
echo "EVO Prediction Job"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Input file: $INPUT_CSV"
echo "Output file: $OUTPUT_CSV"
echo "Model: $MODEL_PATH"
echo "Scaler: $SCALER_PATH"
echo "Start time: $(date)"
echo "=================================================="
echo ""

# Run prediction
python predict_evo.py \
    --input ${INPUT_CSV} \
    --output ${OUTPUT_CSV} \
    --model ${MODEL_PATH} \
    --scaler ${SCALER_PATH} \
    --model_name evo-1.5-8k-base \
    --pooling mean \
    --device cuda

# Check if prediction was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "Prediction completed successfully!"
    echo "Output saved to: $OUTPUT_CSV"
    echo "End time: $(date)"
    echo "=================================================="
else
    echo ""
    echo "=================================================="
    echo "ERROR: Prediction failed!"
    echo "Check the error log for details"
    echo "End time: $(date)"
    echo "=================================================="
    exit 1
fi
