#!/bin/bash
#SBATCH --job-name=evo_predict
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# This sbatch script processes a single CSV file
# Usage: sbatch predict_single.sbatch <input_csv_file> <output_dir> <model_path> <scaler_path>

# Get parameters from command line arguments
INPUT_CSV=$1
OUTPUT_DIR=$2
MODEL_PATH=$3
SCALER_PATH=$4

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

# Record start time for duration calculation
START_TIME=$(date +%s)

# Run prediction
python predict_evo.py \
    --input ${INPUT_CSV} \
    --output ${OUTPUT_CSV} \
    --model ${MODEL_PATH} \
    --scaler ${SCALER_PATH} \
    --model_name evo-1.5-8k-base \
    --pooling mean \
    --device cuda \
    --batch_size 128 \
    --true_batching

# Calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

# Count number of sequences predicted (subtract 1 for header)
if [ -f "$OUTPUT_CSV" ]; then
    NUM_SEQUENCES=$(($(wc -l < "$OUTPUT_CSV") - 1))
else
    NUM_SEQUENCES=0
fi

# Check if prediction was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "Prediction completed successfully!"
    echo "Output saved to: $OUTPUT_CSV"
    echo "Number of sequences: $NUM_SEQUENCES"
    echo "End time: $(date)"
    echo "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s (${DURATION} seconds)"
    echo "=================================================="

    # Record timing to a CSV file for later analysis
    TIMING_LOG="${OUTPUT_DIR}/prediction_timing.csv"

    # Create header if file doesn't exist
    if [ ! -f "$TIMING_LOG" ]; then
        echo "job_id,input_file,num_sequences,start_time,end_time,duration_seconds,status" > "$TIMING_LOG"
    fi

    # Append timing information
    echo "${SLURM_JOB_ID},${BASENAME},${NUM_SEQUENCES},${START_TIME},${END_TIME},${DURATION},success" >> "$TIMING_LOG"

else
    echo ""
    echo "=================================================="
    echo "ERROR: Prediction failed!"
    echo "Check the error log for details"
    echo "End time: $(date)"
    echo "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s (${DURATION} seconds)"
    echo "=================================================="

    # Record failure to timing log
    TIMING_LOG="${OUTPUT_DIR}/prediction_timing.csv"

    # Create header if file doesn't exist
    if [ ! -f "$TIMING_LOG" ]; then
        echo "job_id,input_file,num_sequences,start_time,end_time,duration_seconds,status" > "$TIMING_LOG"
    fi

    # Append timing information for failed job
    echo "${SLURM_JOB_ID},${BASENAME},${NUM_SEQUENCES},${START_TIME},${END_TIME},${DURATION},failed" >> "$TIMING_LOG"

    exit 1
fi
