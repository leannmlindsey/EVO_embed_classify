#!/bin/bash

# Wrapper script to run prediction analysis

SCRIPT_DIR="$(dirname "$0")"

# Check if arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: ./run_metrics_analysis.sh <predictions_dir> <output_csv> [group_by_columns...]"
    echo ""
    echo "Arguments:"
    echo "  predictions_dir:    Directory containing *_predictions.csv files"
    echo "                      (must have 'label' and 'predicted_label' columns)"
    echo "  output_csv:         Output file for metrics results"
    echo "  group_by_columns:   Optional column names to group by (e.g., SeqID)"
    echo ""
    echo "Examples:"
    echo "  # Overall metrics only"
    echo "  ./run_metrics_analysis.sh ./CSV ./results/metrics.csv"
    echo ""
    echo "  # Stratified by sequence ID"
    echo "  ./run_metrics_analysis.sh ./CSV ./results/metrics_by_seqid.csv SeqID"
    echo ""
    echo "  # Stratified by multiple columns"
    echo "  ./run_metrics_analysis.sh ./CSV ./results/metrics.csv SeqID bacterial_phylum"
    exit 1
fi

PREDICTIONS_DIR=$1
OUTPUT_CSV=$2
shift 2
GROUP_BY="$@"

# Check if files/directories exist
if [ ! -d "$PREDICTIONS_DIR" ]; then
    echo "Error: Predictions directory not found: $PREDICTIONS_DIR"
    exit 1
fi

# Create output directory if needed
OUTPUT_DIR=$(dirname "$OUTPUT_CSV")
mkdir -p "$OUTPUT_DIR"

echo "Running prediction analysis..."
echo "Predictions dir: $PREDICTIONS_DIR"
echo "Output: $OUTPUT_CSV"
if [ -n "$GROUP_BY" ]; then
    echo "Grouping by: $GROUP_BY"
fi
echo ""

# Run the analysis
if [ -n "$GROUP_BY" ]; then
    python ${SCRIPT_DIR}/analyze_predictions.py \
        --predictions_dir "$PREDICTIONS_DIR" \
        --output "$OUTPUT_CSV" \
        --group_by $GROUP_BY
else
    python ${SCRIPT_DIR}/analyze_predictions.py \
        --predictions_dir "$PREDICTIONS_DIR" \
        --output "$OUTPUT_CSV"
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "Analysis complete! Results saved to: $OUTPUT_CSV"
else
    echo ""
    echo "Error: Analysis failed. Check the error messages above."
    exit 1
fi
