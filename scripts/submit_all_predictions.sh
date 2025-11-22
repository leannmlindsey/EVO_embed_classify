#!/bin/bash

# Submit prediction jobs for all CSV files in a directory
# Usage: ./submit_all_predictions.sh <data_dir> <output_dir> <model_path> <scaler_path>

# Get the directory where this script is located
SCRIPT_DIR="$(dirname "$0")"
DATA_DIR="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/PHAGE_HOST/EMBEDDINGS/GENOMES/CSV"
OUTPUT_DIR="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/EVO/EVO_GENOME_RESULTS"
MODEL_PATH="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/PHAGE_HOST/EMBEDDINGS/results/nn/evo_nn_classifier.pt"
SCALER_PATH="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/PHAGE_HOST/EMBEDDINGS/results/nn/evo_scaler.joblib"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    exit 1
fi

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

# Check if scaler file exists
if [ ! -f "$SCALER_PATH" ]; then
    echo "Error: Scaler file not found: $SCALER_PATH"
    exit 1
fi

# Create logs and output directories if they don't exist
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# Count CSV files
CSV_COUNT=$(find "$DATA_DIR" -maxdepth 1 -name "*.csv" -type f | wc -l)

if [ $CSV_COUNT -eq 0 ]; then
    echo "Error: No CSV files found in $DATA_DIR"
    exit 1
fi

echo "=================================================="
echo "Submitting EVO prediction jobs"
echo "=================================================="
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Model path: $MODEL_PATH"
echo "Scaler path: $SCALER_PATH"
echo "Number of CSV files: $CSV_COUNT"
echo "Date: $(date)"
echo "=================================================="
echo ""

# Counter for submitted jobs
SUBMITTED=0

# Loop through all CSV files in the directory
for CSV_FILE in "$DATA_DIR"/*.csv; do
    # Check if file exists (handles case where no .csv files exist)
    if [ -f "$CSV_FILE" ]; then
        BASENAME=$(basename "$CSV_FILE")
        echo "Submitting job for: $BASENAME"

        # Submit the job and capture the job ID
        JOB_ID=$(sbatch ${SCRIPT_DIR}/predict_single.sbatch "$CSV_FILE" "$OUTPUT_DIR" "$MODEL_PATH" "$SCALER_PATH" | awk '{print $4}')

        echo "  → Job ID: $JOB_ID"

        SUBMITTED=$((SUBMITTED + 1))

        # Small delay to avoid overwhelming the scheduler
        sleep 0.2
    fi
done

echo ""
echo "=================================================="
echo "Submission complete!"
echo "=================================================="
echo "Total jobs submitted: $SUBMITTED"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: ./logs/"
echo "Predictions will be saved to: $OUTPUT_DIR"
echo "=================================================="
