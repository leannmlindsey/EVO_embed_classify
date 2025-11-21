# EVO Embeddings Classifier

A machine learning pipeline for sequence classification using EVO embeddings. Supports training multiple classifier types (Logistic Regression, SVM, Neural Network) and provides inference capabilities for new sequences.

## Features

- **Multiple Classifier Support**: Train Linear, SVM, or Neural Network classifiers
- **EVO Embeddings**: Extract embeddings from EVO foundation models
- **Flexible Pooling**: Mean or max pooling strategies
- **SLURM Integration**: Ready-to-use SLURM batch scripts for HPC clusters
- **Reproducible**: Random seed support for consistent results
- **Batch Prediction**: Process multiple files in parallel
- **Visualization**: PCA plots and confusion matrices

## Quick Start

### Training

```bash
# Train all three classifiers
python train_evo.py \
    --input_dir ./data \
    --output_dir ./results \
    --embeddings_dir ./embeddings \
    --device cuda \
    --pooling mean

# Train only Neural Network
python train_evo.py \
    --input_dir ./data \
    --output_dir ./results/nn \
    --embeddings_dir ./embeddings \
    --classifiers nn \
    --device cuda \
    --seed 42
```

### Prediction

```bash
# Predict on new sequences
python predict_evo.py \
    --input new_sequences.csv \
    --output predictions.csv \
    --model results/nn/evo_nn_classifier.pt \
    --scaler results/nn/evo_scaler.joblib \
    --device cuda
```

### Batch Prediction

```bash
# Submit all CSV files in a directory for parallel processing
./scripts/submit_all_predictions.sh \
    /path/to/csv/directory \
    /path/to/output \
    /path/to/model.pt \
    /path/to/scaler.joblib

# Example with actual paths
./scripts/submit_all_predictions.sh \
    ./data/to_predict \
    ./predictions \
    ./results/nn_seed42/evo_nn_classifier.pt \
    ./results/nn_seed42/evo_scaler.joblib
```

## Installation

### Requirements

- Python 3.9+
- PyTorch
- scikit-learn
- pandas
- numpy
- tqdm
- matplotlib
- seaborn
- **EVO model** - Must be installed separately (see below)

### EVO Installation

This pipeline uses embeddings from the [EVO foundation model](https://github.com/evo-design/evo). **EVO must be installed on your system before running the embedding and training scripts.**

Follow the installation instructions on the EVO GitHub repository:
- **Repository:** https://github.com/evo-design/evo
- Set up a conda environment according to the directions provided
- Ensure EVO is properly installed and accessible in your environment

### Install Other Dependencies

```bash
pip install torch scikit-learn pandas numpy tqdm matplotlib seaborn
```

## Project Structure

```
.
├── train_evo.py                    # Main training script
├── predict_evo.py                  # Inference script
├── docs/                           # Documentation
│   ├── TRAINING_GUIDE.md          # Training instructions
│   ├── PREDICTION_GUIDE.md        # Single prediction guide
│   ├── BATCH_PREDICTION_GUIDE.md  # Batch prediction guide
│   ├── MULTIPLE_NN_RUNS.md        # Running with multiple seeds
│   ├── SILHOUETTE_GUIDE.md        # Silhouette score analysis
│   └── VERBOSE_OUTPUT.md          # Training output guide
├── scripts/                        # SLURM scripts and utilities
│   ├── train_linear.sbatch        # Train Linear classifier
│   ├── train_svm.sbatch           # Train SVM classifier
│   ├── train_nn.sbatch            # Train NN classifier
│   ├── run_nn_10x.sh              # Run NN with 10 seeds
│   ├── predict_single.sbatch      # Single file prediction
│   ├── submit_all_predictions.sh  # Batch prediction submission
│   ├── concat_metrics.sh          # Concatenate all metrics CSVs
│   ├── analyze_predictions.py     # Comprehensive metrics analysis
│   └── run_metrics_analysis.sh    # Metrics analysis wrapper
└── examples/                       # Example files
    └── example_input.csv          # Example input format
```

## Input Data Format

### Training Data

Training data should be CSV files with:
- `sequence` column: DNA sequences
- `label` column: Binary labels (0 or 1)

```csv
sequence,label
ATCGATCGATCG,0
GCTAGCTAGCTA,1
```

### Prediction Data

Prediction input should be a CSV file with at minimum:
- `sequence` column: DNA sequences to classify

```csv
sequence
ATCGATCGATCG
GCTAGCTAGCTA
```

## Prediction Parameters

When using the SLURM prediction scripts, you need to provide 4 parameters:

1. **Data Directory**: Directory containing CSV files to predict
2. **Output Directory**: Directory where predictions will be saved
3. **Model Path**: Path to trained model checkpoint (`.pt` for NN, `.joblib` for Linear/SVM)
4. **Scaler Path**: Path to the saved scaler file (`.joblib`)

Example:
```bash
./scripts/submit_all_predictions.sh \
    ./data/to_predict \           # Directory with CSV files to process
    ./predictions \                # Output directory
    ./results/nn/evo_nn_classifier.pt \  # Trained model
    ./results/nn/evo_scaler.joblib       # Scaler file
```

## Usage Examples

### Train on SLURM Cluster

```bash
# Submit linear classifier job
sbatch scripts/train_linear.sbatch

# Submit SVM classifier job
sbatch scripts/train_svm.sbatch

# Submit NN classifier job (with seed)
sbatch scripts/train_nn.sbatch 42
```

### Train Multiple NN Models with Different Seeds

```bash
# Automatically submit 10 jobs with seeds 1-10
./scripts/run_nn_10x.sh
```

### Run Predictions on SLURM Cluster

```bash
# Single file prediction
sbatch scripts/predict_single.sbatch \
    /path/to/input.csv \
    /path/to/output \
    /path/to/model.pt \
    /path/to/scaler.joblib

# Batch prediction - submits one job per CSV file in directory
./scripts/submit_all_predictions.sh \
    /path/to/csv/directory \
    /path/to/output \
    /path/to/model.pt \
    /path/to/scaler.joblib
```

### Calculate Silhouette Score Only

```bash
# Skip training, just calculate silhouette and create PCA plot
python train_evo.py \
    --input_dir ./data \
    --output_dir ./results \
    --embeddings_dir ./embeddings \
    --silhouette_only
```

## Output Files

### Training Output

Each training run creates:
- `evo_*_classifier.{joblib,pt}` - Trained model
- `evo_scaler.joblib` - Feature scaler
- `evo_classifier_results_*.csv` - Performance metrics
- `summary_*.txt` - Text summary
- `confusion_matrix_*.png` - Confusion matrix visualization
- `pca_visualization.png` - PCA plot (if silhouette calculated)

### Prediction Output

**Prediction CSV** (`*_predictions.csv`):
- `predicted_class` - Class label (0 or 1)
- `predicted_class_name` - Class name (e.g., "Bacteria" or "Phage")
- `probability` - Probability of class 1
- `confidence` - Model confidence (0.5 to 1.0)

**Metrics CSV** (`*_metrics.csv`):
Each prediction generates a metrics file with:
- `filename` - Input filename
- `n_sequences` - Number of sequences processed
- `TP, TN, FP, FN` - Confusion matrix values
- `accuracy, precision, recall, f1_score, mcc` - Performance metrics
- `mean_probability` - Average prediction probability

**Concatenate All Metrics:**
```bash
# Combine all genome metrics into single file
./scripts/concat_metrics.sh ./predictions all_genomes_metrics.csv
```

### Prediction Analysis & Metrics

After running predictions, analyze results comprehensively:

```bash
# Overall metrics (TP, FP, TN, FN, accuracy, precision, recall, F1, MCC with std devs)
./scripts/run_metrics_analysis.sh \
    ./predictions \
    ./data/test.csv \
    ./results/overall_metrics.csv

# Stratified by bacterial phylum
./scripts/run_metrics_analysis.sh \
    ./predictions \
    ./data/test.csv \
    ./results/metrics_by_phylum.csv \
    bacterial_phylum

# Stratified by multiple metadata columns
./scripts/run_metrics_analysis.sh \
    ./predictions \
    ./data/test.csv \
    ./results/metrics_stratified.csv \
    bacterial_phylum phage_family
```

The analysis script calculates:
- **Confusion Matrix**: TP, FP, TN, FN counts
- **Performance Metrics**: Accuracy, Precision, Recall, F1-score, MCC
- **Standard Deviations**: Bootstrap-based std dev for all metrics (default: 1000 iterations)
- **Stratified Analysis**: Metrics by any metadata column (phylum, family, etc.)

**Requirements**: Ground truth file must have:
- `sequence` column (matching prediction files)
- `label` column (true binary labels: 0 or 1)
- Optional metadata columns for stratified analysis (e.g., `bacterial_phylum`, `phage_family`)

## Configuration

### Model Selection

- `--classifiers linear` - Logistic Regression only
- `--classifiers svm` - SVM only
- `--classifiers nn` - Neural Network only
- `--classifiers all` - All three (default)

### SVM Memory Optimization

For large datasets, use simple mode to reduce memory:

```bash
python train_evo.py --classifiers svm --svm_simple_mode
```

### Embeddings

Embeddings are cached automatically. All jobs can share the same `embeddings_dir`:

```bash
--embeddings_dir ./embeddings
```

## Performance

### Typical Training Times

| Classifier | Device | Time (approx) | Memory |
|------------|--------|---------------|--------|
| Linear     | CPU    | ~30 min       | 32GB   |
| SVM        | CPU    | 2-12 hours    | 256GB  |
| NN         | GPU    | ~5-10 min     | 32GB   |

### Typical Prediction Times

| Sequences | Device | Time (approx) |
|-----------|--------|---------------|
| 100       | GPU    | ~1 min        |
| 1,000     | GPU    | ~5 min        |
| 10,000    | GPU    | ~30 min       |

## Documentation

Detailed guides are available in the `docs/` directory:

- **[Training Guide](docs/TRAINING_GUIDE.md)** - Complete training instructions
- **[Prediction Guide](docs/PREDICTION_GUIDE.md)** - Single file predictions
- **[Batch Prediction Guide](docs/BATCH_PREDICTION_GUIDE.md)** - Parallel processing
- **[Multiple NN Runs](docs/MULTIPLE_NN_RUNS.md)** - Statistical analysis with multiple seeds
- **[Silhouette Guide](docs/SILHOUETTE_GUIDE.md)** - Embedding quality analysis
- **[Verbose Output](docs/VERBOSE_OUTPUT.md)** - Understanding training output

## Citation

If you use this code, please cite the EVO model:

```bibtex
@article{nguyen2024sequence,
   author = {Eric Nguyen and Michael Poli and Matthew G. Durrant and Brian Kang and Dhruva Katrekar and David B. Li and Liam J. Bartie and Armin W. Thomas and Samuel H. King and Garyk Brixi and Jeremy Sullivan and Madelena Y. Ng and Ashley Lewis and Aaron Lou and Stefano Ermon and Stephen A. Baccus and Tina Hernandez-Boussard and Christopher Ré and Patrick D. Hsu and Brian L. Hie },
   title = {Sequence modeling and design from molecular to genome scale with Evo},
   journal = {Science},
   volume = {386},
   number = {6723},
   pages = {eado9336},
   year = {2024},
   doi = {10.1126/science.ado9336},
   URL = {https://www.science.org/doi/abs/10.1126/science.ado9336},
}
```

## License

[Specify license]

## Contact

[Add contact information]

## Troubleshooting

### Out of Memory (OOM)

- For SVM: Use `--svm_simple_mode` flag
- Reduce batch size: `--batch_size 4`
- Use CPU: `--device cpu`

### Slow Training

- Use GPU for Neural Network: `--device cuda`
- Increase batch size: `--batch_size 32`
- Use pre-extracted embeddings via `--embeddings_dir`

### Files Not Found

- Check paths to input files
- Ensure `train.csv`, `dev.csv`, `test.csv` exist in `input_dir`
- Verify model and scaler paths for predictions
- For batch predictions, ensure all 4 parameters are provided:
  - Data directory (containing CSV files to process)
  - Output directory
  - Model path (`.pt` or `.joblib`)
  - Scaler path (`.joblib`)

### Prediction Job Monitoring

```bash
# Check status of submitted jobs
squeue -u $USER

# View prediction logs
tail -f logs/predict_*.out

# Check prediction output
ls -lh predictions/
```

See individual guide documents in `docs/` for more detailed troubleshooting.
