#!/usr/bin/env python3
"""
Simple prediction analysis script.
Calculates metrics (TP, FP, TN, FN, MCC, F1, accuracy, precision, recall)
by comparing predicted_label vs label columns.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)
import warnings
import time
warnings.filterwarnings('ignore')


def calculate_metrics(y_true, y_pred):
    """Calculate all classification metrics including confusion matrix values."""
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        'TP': int(tp),
        'FP': int(fp),
        'TN': int(tn),
        'FN': int(fn),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'mcc': float(mcc),
        'n_samples': len(y_true)
    }


def analyze_predictions(predictions_df, group_by=None):
    """
    Analyze predictions that already contain both true and predicted labels.

    Args:
        predictions_df: DataFrame with both 'label' and 'predicted_label' columns
        group_by: Column name(s) to group by (e.g., 'SeqID', 'bacterial_phylum')

    Returns:
        DataFrame with metrics
    """
    # Ensure we have the required columns
    if 'label' not in predictions_df.columns:
        raise ValueError("predictions_df must have 'label' column")
    if 'predicted_label' not in predictions_df.columns:
        raise ValueError("predictions_df must have 'predicted_label' column")

    results = []

    if group_by is None:
        # Overall metrics
        print(f"   Computing metrics for overall dataset ({len(predictions_df):,} samples)...")
        y_true = predictions_df['label'].values
        y_pred = predictions_df['predicted_label'].values

        metrics = calculate_metrics(y_true, y_pred)
        result = {'group': 'Overall', **metrics}
        results.append(result)

    else:
        # Stratified metrics by group
        if isinstance(group_by, str):
            group_by = [group_by]

        # Check if group columns exist
        missing_cols = [col for col in group_by if col not in predictions_df.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols} in data. Skipping grouped analysis.")
            return pd.DataFrame()

        # Overall metrics first
        print(f"   Computing overall metrics ({len(predictions_df):,} samples)...")
        y_true = predictions_df['label'].values
        y_pred = predictions_df['predicted_label'].values

        metrics = calculate_metrics(y_true, y_pred)
        result = {'group': 'Overall', **metrics}
        results.append(result)

        # Count groups
        grouped = predictions_df.groupby(group_by)
        n_groups = len(grouped)
        print(f"   Computing metrics for {n_groups} groups...")

        # Group-wise metrics
        for i, (group_name, group_df) in enumerate(grouped, 1):
            if len(group_df) < 10:  # Skip very small groups
                continue

            # Format group name
            if isinstance(group_name, tuple):
                group_str = '_'.join(str(g) for g in group_name)
            else:
                group_str = str(group_name)

            print(f"   [{i}/{n_groups}] Processing group: {group_str} ({len(group_df):,} samples)")

            y_true = group_df['label'].values
            y_pred = group_df['predicted_label'].values

            metrics = calculate_metrics(y_true, y_pred)
            result = {'group': group_str, **metrics}
            results.append(result)

    return pd.DataFrame(results)


def collect_all_predictions(predictions_dir):
    """Collect all prediction CSV files from a directory."""
    pred_dir = Path(predictions_dir)

    all_predictions = []

    for pred_file in pred_dir.glob('*_predictions.csv'):
        df = pd.read_csv(pred_file)
        # Add source file info
        df['source_file'] = pred_file.stem.replace('_predictions', '')
        all_predictions.append(df)

    if not all_predictions:
        raise ValueError(f"No prediction files found in {predictions_dir}")

    combined = pd.concat(all_predictions, ignore_index=True)
    print(f"   Collected {len(all_predictions)} prediction files with {len(combined):,} total predictions")

    return combined


def main():
    parser = argparse.ArgumentParser(
        description='Analyze predictions with classification metrics'
    )
    parser.add_argument(
        '--predictions_dir',
        type=str,
        required=True,
        help='Directory containing prediction CSV files (*_predictions.csv) with "label" and "predicted_label" columns'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV file for metrics'
    )
    parser.add_argument(
        '--group_by',
        type=str,
        nargs='+',
        default=None,
        help='Column name(s) to group by (e.g., SeqID bacterial_phylum)'
    )

    args = parser.parse_args()

    start_time = time.time()

    print("=" * 60)
    print("Prediction Analysis")
    print("=" * 60)

    # Collect all predictions (which already contain both labels and predictions)
    print("\n1. Collecting predictions...")
    t0 = time.time()
    predictions = collect_all_predictions(args.predictions_dir)
    print(f"   Time: {time.time() - t0:.2f}s")

    # Verify required columns exist
    if 'label' not in predictions.columns:
        raise ValueError("Prediction files must contain 'label' column")
    if 'predicted_label' not in predictions.columns:
        raise ValueError("Prediction files must contain 'predicted_label' column")

    # Analyze predictions
    print("\n2. Calculating metrics...")
    t0 = time.time()
    results = analyze_predictions(predictions, group_by=args.group_by)
    print(f"   Time: {time.time() - t0:.2f}s")

    # Save results
    print(f"\n3. Saving results to {args.output}...")
    t0 = time.time()
    results.to_csv(args.output, index=False)
    print(f"   Time: {time.time() - t0:.2f}s")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Format for display
    display_cols = ['group', 'n_samples', 'TP', 'FP', 'TN', 'FN',
                    'accuracy', 'precision', 'recall', 'f1_score', 'mcc']

    print("\nMetrics:")
    print(results[display_cols].to_string(index=False))

    print("\n" + "=" * 60)
    print(f"Full results saved to: {args.output}")
    print(f"Total time: {time.time() - start_time:.2f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
