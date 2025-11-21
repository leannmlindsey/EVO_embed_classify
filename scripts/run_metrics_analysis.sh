#!/bin/bash

# Wrapper script to run comprehensive prediction analysis

SCRIPT_DIR="$(dirname "$0")"

# Check if arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: ./run_metrics_analysis.sh <predictions_dir> <ground_truth_csv> <output_csv> [group_by_columns...]"
    echo ""
    echo "Arguments:"
    echo "  predictions_dir:    Directory containing *_predictions.csv files"
    echo "  ground_truth_csv:   CSV file with ground truth labels (must have 'sequence' and 'label' columns)"
    echo "  output_csv:         Output file for metrics results"
    echo "  group_by_columns:   Optional column names to group by (e.g., bacterial_phylum phage_phylum)"
    echo ""
    echo "Examples:"
    echo "  # Overall metrics only"
    echo "  ./run_metrics_analysis.sh ./predictions ./data/test.csv ./results/metrics.csv"
    echo ""
    echo "  # Stratified by bacterial phylum"
    echo "  ./run_metrics_analysis.sh ./predictions ./data/test.csv ./results/metrics_by_phylum.csv bacterial_phylum"
    echo ""
    echo "  # Stratified by multiple columns"
    echo "  ./run_metrics_analysis.sh ./predictions ./data/test.csv ./results/metrics_stratified.csv bacterial_phylum phage_family"
    exit 1
fi

PREDICTIONS_DIR=$1
GROUND_TRUTH=$2
OUTPUT_CSV=$3
shift 3
GROUP_BY="$@"

# Check if files/directories exist
if [ ! -d "$PREDICTIONS_DIR" ]; then
    echo "Error: Predictions directory not found: $PREDICTIONS_DIR"
    exit 1
fi

if [ ! -f "$GROUND_TRUTH" ]; then
    echo "Error: Ground truth file not found: $GROUND_TRUTH"
    exit 1
fi

# Create output directory if needed
OUTPUT_DIR=$(dirname "$OUTPUT_CSV")
mkdir -p "$OUTPUT_DIR"

echo "Running comprehensive prediction analysis..."
echo "Predictions dir: $PREDICTIONS_DIR"
echo "Ground truth: $GROUND_TRUTH"
echo "Output: $OUTPUT_CSV"
if [ -n "$GROUP_BY" ]; then
    echo "Grouping by: $GROUP_BY"
fi
echo ""

# Run the analysis
if [ -n "$GROUP_BY" ]; then
    python ${SCRIPT_DIR}/analyze_predictions.py \
        --predictions_dir "$PREDICTIONS_DIR" \
        --ground_truth "$GROUND_TRUTH" \
        --output "$OUTPUT_CSV" \
        --group_by $GROUP_BY
else
    python ${SCRIPT_DIR}/analyze_predictions.py \
        --predictions_dir "$PREDICTIONS_DIR" \
        --ground_truth "$GROUND_TRUTH" \
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
