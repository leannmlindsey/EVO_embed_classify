#!/bin/bash

# Submit prediction jobs for all CSV files in a directory
# Usage: ./submit_all_predictions.sh <directory_with_csv_files>

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if directory argument is provided
if [ -z "$1" ]; then
    echo "Error: No directory provided"
    echo "Usage: ./submit_all_predictions.sh <directory_with_csv_files>"
    exit 1
fi

INPUT_DIR=$1

# Check if directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory not found: $INPUT_DIR"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Count CSV files
CSV_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -name "*.csv" -type f | wc -l)

if [ $CSV_COUNT -eq 0 ]; then
    echo "Error: No CSV files found in $INPUT_DIR"
    exit 1
fi

echo "=================================================="
echo "Submitting EVO prediction jobs"
echo "=================================================="
echo "Input directory: $INPUT_DIR"
echo "Number of CSV files: $CSV_COUNT"
echo "Date: $(date)"
echo "=================================================="
echo ""

# Counter for submitted jobs
SUBMITTED=0

# Loop through all CSV files in the directory
for CSV_FILE in "$INPUT_DIR"/*.csv; do
    # Check if file exists (handles case where no .csv files exist)
    if [ -f "$CSV_FILE" ]; then
        BASENAME=$(basename "$CSV_FILE")
        echo "Submitting job for: $BASENAME"

        # Submit the job and capture the job ID
        JOB_ID=$(sbatch ${SCRIPT_DIR}/predict_single.sbatch "$CSV_FILE" | awk '{print $4}')

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
echo "Predictions will be saved to: ./predictions/"
echo "=================================================="
