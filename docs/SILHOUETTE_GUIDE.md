# Calculating Silhouette Scores Only

If you already have embeddings and just want to calculate silhouette scores without retraining classifiers, use the `--silhouette_only` flag.

## Usage

```bash
python train_evo.py \
    --input_dir ./data \
    --output_dir ./results/linear \
    --embeddings_dir ./embeddings \
    --pooling mean \
    --silhouette_only
```

## What It Does

1. **Loads pre-extracted embeddings** from `embeddings_dir`
2. **Calculates two silhouette scores:**
   - Original high-dimensional (4096D) space
   - PCA 2D projection
3. **Creates PCA visualization** (`pca_visualization.png`)
4. **Saves scores** to `silhouette_scores_mean.txt`
5. **Exits** without training any classifiers

## Output Files

- `silhouette_scores_mean.txt` - Text file with both scores
- `pca_visualization.png` - Scatter plot showing bacteria vs phage in PCA space

## Example Output

```
Calculating silhouette scores...
Silhouette Score (Original 4096D): 0.XXXX
Performing PCA (2 components) for visualization and silhouette...
Silhouette Score (PCA 2D): 0.XXXX
PCA explained variance: 0.XXXX (0.XXXX + 0.XXXX)
PCA visualization saved to results/linear/pca_visualization.png

==================================================
Silhouette-only mode: Skipping classifier training
==================================================

Silhouette scores saved to results/linear/silhouette_scores_mean.txt
PCA visualization saved to pca_visualization.png

Done! Exiting without training classifiers.
```

## Notes

- **Fast**: Takes only seconds vs hours for full training
- **Requires embeddings**: Embeddings must already exist in `embeddings_dir`
- **No classifier needed**: Does not require trained models
- **Can run on existing results**: Point `--output_dir` to where you want the PCA plot saved
