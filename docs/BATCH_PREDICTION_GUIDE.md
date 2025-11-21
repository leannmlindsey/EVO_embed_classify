# Batch Prediction Guide

Process multiple CSV files in parallel using SLURM job submission.

## Quick Start

```bash
# Submit all CSV files in a directory
./submit_all_predictions.sh /path/to/csv/directory
```

This will:
- Find all `.csv` files in the directory
- Submit each as a separate SLURM job
- Process all files in parallel
- Save predictions to `./predictions/` with unique names

## Files Overview

### 1. `predict_single.sbatch`
- SLURM job script for processing **one** CSV file
- Takes input filename as argument
- Automatically names output file to avoid overwrites

### 2. `submit_all_predictions.sh`
- Bash script that submits jobs for **all** CSV files in a directory
- Loops through directory and calls `predict_single.sbatch` for each file

## Directory Structure

### Before Running
```
your_data_directory/
├── sample_001.csv
├── sample_002.csv
├── sample_003.csv
└── ... (75 more files)
```

### After Running
```
predictions/
├── sample_001_predictions.csv
├── sample_002_predictions.csv
├── sample_003_predictions.csv
└── ... (75 more files)

logs/
├── predict_12345.out  (stdout for job 12345)
├── predict_12345.err  (stderr for job 12345)
├── predict_12346.out
├── predict_12346.err
└── ...
```

## Usage

### Step 1: Prepare Your Files

Make sure you have:
- ✅ Directory with 78 CSV files
- ✅ Each CSV has a `sequence` column
- ✅ Trained model at `./results/nn_seed1/evo_nn_classifier.pt`
- ✅ Scaler at `./results/nn_seed1/evo_scaler.joblib`

### Step 2: Update Paths (if needed)

Edit `predict_single.sbatch` if your model/scaler are in different locations:

```bash
MODEL_PATH="./results/nn_seed1/evo_nn_classifier.pt"  # Update this
SCALER_PATH="./results/nn_seed1/evo_scaler.joblib"    # Update this
```

### Step 3: Submit All Jobs

```bash
./submit_all_predictions.sh /path/to/your/csv/directory
```

Example:
```bash
./submit_all_predictions.sh ./data/sequences_to_predict/
```

### Step 4: Monitor Jobs

Check job status:
```bash
squeue -u $USER
```

Check how many are running/pending:
```bash
squeue -u $USER | grep evo_predict | wc -l
```

### Step 5: Check Results

View output of a specific job:
```bash
tail -f logs/predict_12345.out
```

Check for errors:
```bash
grep -l "ERROR" logs/predict_*.err
```

Count completed predictions:
```bash
ls predictions/*.csv | wc -l
```

## Output File Naming

Input files are automatically renamed to avoid overwrites:

| Input File | Output File |
|------------|-------------|
| `sample_001.csv` | `predictions/sample_001_predictions.csv` |
| `sample_002.csv` | `predictions/sample_002_predictions.csv` |
| `data.csv` | `predictions/data_predictions.csv` |

## Resource Requirements

Each job requests:
- **1 GPU** (A100 or similar)
- **8 CPUs**
- **32 GB RAM**
- **2 hours** time limit

Adjust in `predict_single.sbatch` if needed:
```bash
#SBATCH --cpus-per-task=8      # Increase if needed
#SBATCH --mem=32G              # Increase if OOM
#SBATCH --time=02:00:00        # Increase for large files
```

## Performance Estimates

Approximate processing times (per file):

| Sequences | Time (GPU) | Time (CPU) |
|-----------|------------|------------|
| 100 | ~1 min | ~5 min |
| 1,000 | ~5 min | ~30 min |
| 10,000 | ~30 min | ~4 hours |

**78 files in parallel** will complete in the time of your slowest file!

## Troubleshooting

### Problem: Jobs stay in pending state

**Check queue limits:**
```bash
squeue -u $USER
```

Your cluster may have limits on:
- Number of concurrent jobs
- Number of GPUs per user

**Solution:** Submit in batches if needed

### Problem: Some jobs fail with OOM

**Check error logs:**
```bash
grep "out of memory" logs/predict_*.err
```

**Solution:** Edit `predict_single.sbatch`:
```bash
#SBATCH --mem=64G  # Increase from 32G
```

Or reduce batch size in the script:
```bash
--batch_size 4  # Reduce from 8
```

### Problem: Wrong model/scaler path

**Check paths in sbatch file:**
```bash
grep "MODEL_PATH\|SCALER_PATH" predict_single.sbatch
```

Make sure they point to existing files.

### Problem: Output files missing

**Check if jobs completed:**
```bash
sacct -u $USER --format=JobID,JobName,State,ExitCode | grep evo_predict
```

**Check specific job log:**
```bash
tail logs/predict_JOBID.err
```

## Advanced Usage

### Submit Subset of Files

Instead of all files:
```bash
for file in ./data/sample_00{1..10}.csv; do
    sbatch predict_single.sbatch "$file"
done
```

### Process with Different Model

Edit `predict_single.sbatch`:
```bash
MODEL_PATH="./results/nn_seed5/evo_nn_classifier.pt"  # Use different seed
```

### Change Output Directory

Edit `predict_single.sbatch`:
```bash
OUTPUT_DIR="./predictions_seed5"  # Different output location
```

### Run on CPU Instead of GPU

Edit `predict_single.sbatch`:
```bash
#SBATCH --partition=norm  # Change from gpu
# Remove: #SBATCH --gres=gpu:1

# In the python command:
--device cpu  # Change from cuda
```

## Collecting Results

After all jobs complete, combine results:

```bash
# Create header from first file
head -1 predictions/sample_001_predictions.csv > all_predictions.csv

# Append all predictions (skip headers)
for file in predictions/*_predictions.csv; do
    tail -n +2 "$file" >> all_predictions.csv
done

echo "Combined $(wc -l < all_predictions.csv) total predictions"
```

## Example Workflow

```bash
# 1. Check your data
ls data/sequences/*.csv | wc -l
# Output: 78

# 2. Submit all jobs
./submit_all_predictions.sh data/sequences/
# Output: Total jobs submitted: 78

# 3. Monitor progress
watch -n 10 'squeue -u $USER | grep evo_predict | wc -l'

# 4. Check completion
ls predictions/*.csv | wc -l
# Should eventually be 78

# 5. Combine results
head -1 predictions/$(ls predictions/ | head -1) > all_predictions.csv
tail -n +2 -q predictions/*.csv >> all_predictions.csv

# 6. Summary statistics
echo "Total predictions: $(tail -n +2 all_predictions.csv | wc -l)"
echo "Phage predictions: $(tail -n +2 all_predictions.csv | grep ',1,Phage,' | wc -l)"
echo "Bacteria predictions: $(tail -n +2 all_predictions.csv | grep ',0,Bacteria,' | wc -l)"
```
