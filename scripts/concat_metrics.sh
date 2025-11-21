#!/bin/bash

# Concatenate all metrics CSV files from predictions
# Usage: ./concat_metrics.sh <results_directory> <output_file>

if [ $# -lt 2 ]; then
    echo "Usage: ./concat_metrics.sh <results_directory> <output_file>"
    echo ""
    echo "Example:"
    echo "  ./concat_metrics.sh /path/to/predictions all_metrics.csv"
    exit 1
fi

RESULTS_DIR=$1
OUTPUT_FILE=$2

# Check if directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Directory not found: $RESULTS_DIR"
    exit 1
fi

# Count metrics files
METRICS_COUNT=$(find "$RESULTS_DIR" -name "*_metrics.csv" -type f | wc -l)

if [ $METRICS_COUNT -eq 0 ]; then
    echo "Error: No *_metrics.csv files found in $RESULTS_DIR"
    exit 1
fi

echo "Found $METRICS_COUNT metrics files"
echo "Concatenating to $OUTPUT_FILE..."

# Get the first file to use as header
FIRST_FILE=$(find "$RESULTS_DIR" -name "*_metrics.csv" -type f | head -1)

# Write header from first file
head -1 "$FIRST_FILE" > "$OUTPUT_FILE"

# Append all files (skip headers)
find "$RESULTS_DIR" -name "*_metrics.csv" -type f | while read file; do
    tail -n +2 "$file" >> "$OUTPUT_FILE"
done

echo "✓ Concatenated $METRICS_COUNT files to $OUTPUT_FILE"

# Calculate summary statistics
echo ""
echo "Summary Statistics:"
python3 << EOF
import pandas as pd
df = pd.read_csv('$OUTPUT_FILE')
print(f"Total genomes: {len(df)}")
print(f"Total sequences: {df['n_sequences'].sum()}")
print("\nMean metrics across all genomes:")
print(df[['accuracy', 'precision', 'recall', 'f1_score', 'mcc']].mean().to_string())
print("\nStandard deviation:")
print(df[['accuracy', 'precision', 'recall', 'f1_score', 'mcc']].std().to_string())
EOF
