#!/bin/bash
#SBATCH --output=logs/%x_%j.out  # Redirects outputs to file in current_dir/logs
#SBATCH --error=logs/%x_%j.out  # Redirects err to same file in current_dir/logs
#SBATCH --job-name=fsdp_smp

# has to be shared dir
CONDA_ENV_PATH=${1:-"$CONDA_DEFAULT_ENV"}
SHELL_SCRIPT=${2:-"scripts/model.sh"}
shift 2
SCRIPT_ARGS=$@

if [ -z $CONDA_ENV_PATH ]; then
    echo "Conda env path needs to be passed. Exiting"
    exit 1
fi
if [ -z "$SCRIPT_ARGS" ]; then
    SCRIPT_ARGS=""
else
    SCRIPT_ARGS+=" "
fi

HOSTFILE=hosts_${SLURM_JOB_ID}
scontrol show hostnames | sort > $HOSTFILE
srun -l -D `pwd` conda run -p $CONDA_ENV_PATH --no-capture-output $SHELL_SCRIPT --hostfile $HOSTFILE $SCRIPT_ARGS