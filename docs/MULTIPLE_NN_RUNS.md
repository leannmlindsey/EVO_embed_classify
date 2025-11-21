# Running Multiple NN Jobs with Different Seeds

To run the neural network classifier 10 times with different random seeds for statistical analysis:

## Using the Provided Script (Recommended)

Simply run:
```bash
./run_nn_10x.sh
```

This will submit 10 jobs with seeds 1-10, saving results to:
- `./results/nn_seed1/`
- `./results/nn_seed2/`
- ...
- `./results/nn_seed10/`

## Manual Submission

You can also manually submit individual runs:

```bash
sbatch train_nn.sbatch 1
sbatch train_nn.sbatch 2
sbatch train_nn.sbatch 3
# ... etc
```

Each job will automatically save to `./results/nn_seed${SEED}/`

## Monitoring All Jobs

Check status of all your NN jobs:
```bash
squeue -u $USER | grep evo_nn
```

View output from a specific seed:
```bash
tail -f logs/nn_*.out
```

## Collecting Results

After all jobs complete, you can aggregate results:

```bash
# Create a summary CSV with results from all 10 runs
echo "Seed,Accuracy,Precision,Recall,F1,MCC,ROC_AUC,Training_Time" > nn_all_seeds_results.csv

for seed in {1..10}; do
    # Extract metrics from each results CSV and append to summary
    tail -n 1 ./results/nn_seed${seed}/evo_classifier_results_mean.csv >> nn_all_seeds_results.csv
done
```

Then calculate mean ± std in Python/R for your paper.

## Important Notes

1. **All jobs share the same embeddings** - They all use `./embeddings/` directory
2. **Jobs can run in parallel** - Since they write to different output directories (nn_seed1, nn_seed2, etc.)
3. **Random seeds ensure reproducibility** - You can recreate any specific run
4. **The seed is passed as an argument** - `train_nn.sbatch` takes the seed as $1
