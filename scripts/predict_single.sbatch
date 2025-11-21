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

# Get input filename from command line argument
INPUT_CSV=$1
DATA_DIR=$2
OUTPUT_DIR=$3
MODEL_PATH=$4
SCALER_PATH=$5

# Get seed from command line argument
SEED=$1

script_dir="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/EVO/EVO_embed_classify"

cd ${script_dir}

# Setup environment
module load cuda
nvidia-smi
source activate evo

# Create logs directory if it doesn't exist
mkdir -p logs


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
