## SMP v2 Accelerate Examples
In this directory we have example scripts for training with Accelerate. We assume you have already setup a conda environment with SMP Pytorch. Below we first describe the files in this directory, and then go over how to run some jobs.

### Files
- `run_clm_no_trainer.py` : Entrypoint script. 
- `scripts/model.sh` : Main script which passes the config and launches `run_clm_no_trainer.py`. This is used by `conda_launch.sh` and scripts in convergence_jobs folder. If you want to define your own model configuration you might want to modify this.

#### Launch scripts
- `conda_launch.sh` : This is a slurm script which launches a job using the activated conda environment. It expects to be run on the master node of the Slurm cluster. See below section for instructions. By default it runs with synthetic data to make it easy to test the scripts.
## Note on paths
These scripts need to be put on a directory that can be accessed on all nodes, such as FSX.
We also recommend setting all paths (for input data and checkpoints) as shared directories using FSX.
These paths can be set in scripts as shown in `convergence_jobs/neox_7b/neox_7b_4Mtokens.sh`.

## User Guide

1. Launching a job with Llama v2 on 4 nodes. The default config in the script loads a pre-trained Llama v2 7b model and finetunes it.
```
sbatch -N 4 conda_launch.sh /PATH/TO/CONDA/ENV
```

2. Changing arguments taken by the script.
`model.sh` takes certain arguments from the launch script, and uses them to pass args to the training script. You can refer to `model.sh` if those are the arguments you would like to change. 

3. Running with another model.
If you want to run with another Huggingface model thats non-default, first you should download the Huggingface model repository to your local FSx, via the `utils/download_hf_hub_repo.py`. Update the parameters within it and then run `python utils/download_hf_hub_repo.py`. Then, you will need to update `model.sh` to set the appropriate path, and `run_clm_no_trainer.py` to add the new transformer layer as a supported model.
