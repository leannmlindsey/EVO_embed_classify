#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Submit 10 NN jobs with seeds 1-10
for seed in {1..10}; do
    sbatch ${SCRIPT_DIR}/train_nn.sbatch ${seed}
done
