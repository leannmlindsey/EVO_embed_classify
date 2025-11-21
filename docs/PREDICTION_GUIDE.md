# EVO Sequence Prediction Guide

Use `predict_evo.py` to classify new sequences using your trained neural network classifier.

## Quick Start

```bash
python predict_evo.py \
    --input new_sequences.csv \
    --output predictions.csv \
    --model results/nn_seed1/evo_nn_classifier.pt \
    --scaler results/nn_seed1/evo_scaler.joblib \
    --model_name evo-1.5-8k-base \
    --pooling mean \
    --device cuda
```

## Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--input` | Input CSV file with sequences | `new_data.csv` |
| `--output` | Output CSV file for predictions | `predictions.csv` |
| `--model` | Path to trained NN model (.pt) | `results/nn_seed1/evo_nn_classifier.pt` |
| `--scaler` | Path to fitted scaler (.joblib) | `results/nn_seed1/evo_scaler.joblib` |

## Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `evo-1` | EVO model name (must match training) |
| `--pooling` | `mean` | Pooling strategy: `mean` or `max` |
| `--batch_size` | `8` | Batch size for embedding extraction |
| `--device` | `cuda` (if available) | Device: `cuda` or `cpu` |
| `--sequence_col` | `sequence` | Name of sequence column |
| `--index_col` | `None` | Name of index column (optional) |
| `--class_names` | `Bacteria Phage` | Names for class 0 and class 1 |

## Input CSV Format

Your input CSV must have at least one column with DNA sequences.

### Minimal Format
```csv
sequence
ATCGATCGATCG
GCTAGCTAGCTA
TTAATTAATTAA
```

### With Index Column
```csv
seq_id,sequence
seq_001,ATCGATCGATCG
seq_002,GCTAGCTAGCTA
seq_003,TTAATTAATTAA
```

### With Additional Metadata
```csv
seq_id,sequence,organism,length
seq_001,ATCGATCGATCG,Unknown,12
seq_002,GCTAGCTAGCTA,Unknown,12
seq_003,TTAATTAATTAA,Unknown,12
```

## Output CSV Format

The script adds four columns to your input data:

| Column | Description |
|--------|-------------|
| `predicted_class` | Predicted class (0 or 1) |
| `predicted_class_name` | Class name (e.g., "Bacteria" or "Phage") |
| `probability` | Probability of being class 1 (Phage) |
| `confidence` | Maximum probability (how confident the model is) |

### Example Output
```csv
seq_id,sequence,predicted_class,predicted_class_name,probability,confidence
seq_001,ATCGATCGATCG,1,Phage,0.XXXX,0.XXXX
seq_002,GCTAGCTAGCTA,0,Bacteria,0.XXXX,0.XXXX
seq_003,TTAATTAATTAA,1,Phage,0.XXXX,0.XXXX
```

## Example Usage

### Basic Prediction
```bash
python predict_evo.py \
    --input new_sequences.csv \
    --output predictions.csv \
    --model results/nn_seed1/evo_nn_classifier.pt \
    --scaler results/nn_seed1/evo_scaler.joblib
```

### With Custom Column Names
```bash
python predict_evo.py \
    --input data.csv \
    --output predictions.csv \
    --model results/nn_seed1/evo_nn_classifier.pt \
    --scaler results/nn_seed1/evo_scaler.joblib \
    --sequence_col dna_sequence \
    --index_col id
```

### With Custom Class Names
```bash
python predict_evo.py \
    --input sequences.csv \
    --output predictions.csv \
    --model results/nn_seed1/evo_nn_classifier.pt \
    --scaler results/nn_seed1/evo_scaler.joblib \
    --class_names Host Virus
```

### Using Different EVO Model (must match training!)
```bash
python predict_evo.py \
    --input sequences.csv \
    --output predictions.csv \
    --model results/nn_seed1/evo_nn_classifier.pt \
    --scaler results/nn_seed1/evo_scaler.joblib \
    --model_name evo-1.5-8k-base \
    --pooling mean
```

### CPU-only Prediction
```bash
python predict_evo.py \
    --input sequences.csv \
    --output predictions.csv \
    --model results/nn_seed1/evo_nn_classifier.pt \
    --scaler results/nn_seed1/evo_scaler.joblib \
    --device cpu
```

## SLURM Job Script Example

Create `predict.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=evo_predict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/predict_%j.out
#SBATCH --error=logs/predict_%j.err

# Load modules
module load python/3.9
module load cuda/11.7

# Activate environment
source activate evo

# Set paths
INPUT_CSV="new_sequences.csv"
OUTPUT_CSV="predictions.csv"
MODEL_PATH="results/nn_seed1/evo_nn_classifier.pt"
SCALER_PATH="results/nn_seed1/evo_scaler.joblib"

# Run prediction
python predict_evo.py \
    --input ${INPUT_CSV} \
    --output ${OUTPUT_CSV} \
    --model ${MODEL_PATH} \
    --scaler ${SCALER_PATH} \
    --model_name evo-1.5-8k-base \
    --pooling mean \
    --device cuda

echo "Prediction completed!"
```

Submit with:
```bash
sbatch predict.sbatch
```

## Important Notes

### Model Compatibility
**CRITICAL**: The prediction settings must match your training settings:

- `--model_name` must be the same EVO model used during training
- `--pooling` must be the same pooling strategy used during training
- Use the scaler from the same training run as the model

### File Locations
- Model files are in `results/nn_seed*/evo_nn_classifier.pt`
- Scaler files are in `results/nn_seed*/evo_scaler.joblib`
- All 10 seed models should give similar predictions (use best performing one)

### Performance
- GPU is **much faster** for large datasets
- Processing time depends on:
  - Number of sequences
  - Sequence length
  - Batch size (increase for faster processing)
  - Device (GPU vs CPU)

### Memory Requirements
- Adjust `--batch_size` if you run out of memory
- Typical: 8-32 for GPU, 1-4 for CPU
- Larger batch = faster but more memory

## Expected Output

```
Using device: cuda
Loading input file: new_sequences.csv
Loaded 1000 sequences
Loading EVO model: evo-1.5-8k-base...
Applying monkey patch to obtain embeddings...
Extracting embeddings for 1000 sequences...
100%|████████████████████| 125/125 [00:15<00:00,  8.12it/s]
Embeddings extracted. Shape: (1000, 4096)
Loading scaler from results/nn_seed1/evo_scaler.joblib...
Loading trained model from results/nn_seed1/evo_nn_classifier.pt...
Model loaded successfully
Scaling embeddings...
Making predictions...
Preparing output...
Saving predictions to predictions.csv...

==================================================
Prediction Summary
==================================================
Total sequences: 1000

Class distribution:
  Bacteria (class 0): 423 (42.3%)
  Phage (class 1): 577 (57.7%)

Average confidence: 0.8734
Min confidence: 0.5123
Max confidence: 0.9998
==================================================

Predictions saved to: predictions.csv
```

## Troubleshooting

### Error: "Column 'sequence' not found"
- Specify your sequence column name with `--sequence_col your_column_name`

### Error: "Model file not found"
- Check the path to your .pt file
- Make sure you trained a model first

### Error: "Out of memory"
- Reduce `--batch_size` (try 4, 2, or 1)
- Use CPU instead of GPU with `--device cpu`

### Predictions seem wrong
- Verify `--model_name` and `--pooling` match your training
- Make sure you're using the correct scaler file
- Check that sequences are DNA (not RNA or protein)
