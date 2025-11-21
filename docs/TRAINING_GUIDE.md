# SBATCH Job Submission Guide

This directory contains three SBATCH scripts for training individual classifiers on Biowulf.

## Files

1. **train_linear.sbatch** - Linear (Logistic Regression) classifier
2. **train_svm.sbatch** - SVM classifier
3. **train_nn.sbatch** - Neural Network classifier

## Before Submitting

### 1. Update paths in each sbatch file:

```bash
INPUT_DIR="./data"  # Path to directory containing train.csv, dev.csv, test.csv
OUTPUT_DIR="./results/linear"  # Where to save results (different for each job)
EMBEDDINGS_DIR="./embeddings"  # Shared directory for embeddings (SAME for all jobs)
```

**Important:** All three scripts share the same `EMBEDDINGS_DIR` so embeddings are only extracted/saved once!

### 2. Adjust resource requirements if needed:

**Linear Classifier (`train_linear.sbatch`):**
- Partition: `norm` (CPU only)
- CPUs: 16
- Memory: 32GB
- Time: 2 hours
- Expected runtime: ~30-60 minutes

**SVM Classifier (`train_svm.sbatch`):**
- Partition: `norm` (CPU only)
- CPUs: 8 (reduced to avoid memory issues with parallel workers)
- Memory: 256GB (increased for large datasets)
- Time: 48 hours
- Expected runtime: Variable (could be 4-24+ hours depending on dataset size)
- **Note:** Now uses `--svm_simple_mode` flag by default
  - Simple mode: Trains with fixed C=1.0, no hyperparameter search (minimal memory)
  - To enable full hyperparameter search: Remove `--svm_simple_mode` flag (requires more memory)

**Neural Network (`train_nn.sbatch`):**
- Partition: `gpu`
- GPUs: 1
- CPUs: 8
- Memory: 32GB
- Time: 12 hours
- Expected runtime: ~2-4 hours

### 3. Create logs directory:

```bash
mkdir -p logs
```

## Submitting Jobs

Submit all three jobs at once:
```bash
sbatch train_linear.sbatch
sbatch train_svm.sbatch
sbatch train_nn.sbatch
```

Or submit individually as needed.

## Monitoring Jobs

Check job status:
```bash
squeue -u $USER
```

View live output:
```bash
tail -f logs/linear_<jobid>.out
tail -f logs/svm_<jobid>.out
tail -f logs/nn_<jobid>.out
```

Check for errors:
```bash
tail -f logs/linear_<jobid>.err
tail -f logs/svm_<jobid>.err
tail -f logs/nn_<jobid>.err
```

## Output Files

Each job will create a separate output directory:

**Linear Classifier** (`results/linear/`):
- `evo_linear_classifier.joblib` - Trained model
- `evo_scaler.joblib` - Feature scaler
- `evo_classifier_results_mean.csv` - Metrics
- `summary_mean.txt` - Summary report
- `confusion_matrix_Logistic_Regression.png` - Confusion matrix

**SVM Classifier** (`results/svm/`):
- `evo_svm_classifier.joblib` - Trained model
- `evo_scaler.joblib` - Feature scaler
- `evo_classifier_results_mean.csv` - Metrics
- `summary_mean.txt` - Summary report
- `confusion_matrix_SVM_(Linear_Kernel).png` - Confusion matrix

**Neural Network** (`results/nn/`):
- `evo_nn_classifier.pt` - Trained model
- `evo_scaler.joblib` - Feature scaler
- `evo_classifier_results_mean.csv` - Metrics
- `summary_mean.txt` - Summary report
- `confusion_matrix_3-Layer_Neural_Network.png` - Confusion matrix

## Notes

- **Embeddings are shared:** All three jobs use the same `EMBEDDINGS_DIR` (default: `./embeddings/`)
- If embeddings don't exist, they will be extracted once and saved to `EMBEDDINGS_DIR`
- Embeddings will look for these files in `EMBEDDINGS_DIR`:
  - `train_embeddings_mean.npz`
  - `dev_embeddings_mean.npz`
  - `test_embeddings_mean.npz`
- Put your pre-extracted embeddings in `./embeddings/` to avoid re-extraction
- All three jobs can run simultaneously and share the same embeddings
- The SVM job may take significantly longer than the others
- Log files are saved with job IDs for easy tracking

## Random Seeds for Reproducibility

The `train_evo.py` script now supports a `--seed` argument for reproducible results:

- **Neural Network**: Recommended to run 10+ times with different seeds to get mean ± std
- See `MULTIPLE_NN_RUNS.md` for instructions on running multiple NN jobs with different seeds
- To run a single NN job with a specific seed: `sbatch train_nn.sbatch 42`
- To run 10 NN jobs with different seeds: `./run_nn_10x.sh`
