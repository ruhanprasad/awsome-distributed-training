#!/bin/bash
#SBATCH --job-name=datapipeline_tests
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --nodes=1

CONDA_ENV_PATH=${1:-"$CONDA_DEFAULT_ENV"}
export PYTHONPATH=.

# RUN from transformers_fsdp dir

if [[ -z "${CONDA_ENV_PATH}" ]]; then
    echo "Conda env not set, exiting"
fi
conda run -p $CONDA_ENV_PATH --no-capture-output torchrun --nnodes=1 --nproc_per_node=8 tests/data/test_data_pipeline.py
