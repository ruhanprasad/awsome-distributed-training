#!/bin/bash
#SBATCH --output=logs/%x_%j.out  # Redirects outputs to file in current_dir/logs
#SBATCH --error=logs/%x_%j.out  # Redirects err to same file in current_dir/logs
#SBATCH --job-name=gpt2_7b_hf


# Example usage
## Docker
### sbatch -N 4 --exclude=compute-44 convergence_jobs/gpt2_7b/gpt2_7b_4Mtokens_fsdp.sh -i XXX
### sbatch -N 4 --exclude=compute-44 convergence_jobs/gpt2_7b/gpt2_7b_4Mtokens_fsdp.sh (uses default image)
## Conda
### sbatch -N 4 --exclude=compute-44 convergence_jobs/gpt2_7b/gpt2_7b_4Mtokens_fsdp.sh (uses activated conda env)

parse_inputs() {
    MODE="docker"
    ENV_PATH=$CONDA_DEFAULT_ENV
    CONTAINER_IMAGE="855988369404.dkr.ecr.us-west-2.amazonaws.com/sm-pytorch-conda-builder:sm-pytorch_sm-v2.0.1"
    while [[ $# -gt 0 ]]; do
        key="$1"
        case $key in
        -env)
            ENV_PATH=$2
            MODE="conda"
            shift 2
            ;;
        -i | --image)
            MODE="docker"
            CONTAINER_IMAGE=$2
            shift 2
            ;;
        *)
            break
            ;;
        esac
    done
}

parse_inputs $@

## DATA
SCRIPT_ARGS="--training_dir /fsx/datasets/train_ids_wsvocab_redo_2048_smaller "
SCRIPT_ARGS+="--test_dir /fsx/datasets/val_ids_wsvocab_2048 "

## MODEL
SCRIPT_ARGS+="--model_type gpt2 --model_size 7b "
# SCRIPT_ARGS+="--use_smp_implementation 1 --activation_checkpointing 0 --tensor_parallel_degree 1 "
max_context_width=2048  # seqlen

## BATCH SIZE
# if [ $NUM_NODES -lt 16 ]; then
#     echo "Can't use 4M tokens with less than 16 nodes"
#     exit 1
# else
#     GLOBAL_BATCH_SIZE=4194304
# fi
# train_batch_size=$(python -c "print($GLOBAL_BATCH_SIZE//($NUM_NODES * 8 * $max_context_width))")
train_batch_size=2

SCRIPT_ARGS+="--train_batch_size $train_batch_size "
SCRIPT_ARGS+="--val_batch_size $train_batch_size "
SCRIPT_ARGS+="--max_context_width $max_context_width "
SCRIPT_ARGS+="--max_steps 143000 "
SCRIPT_ARGS+="--validation_freq 100 "

## ARTIFACTS
SCRIPT_ARGS+="--checkpoint_dir checkpoints/$SLURM_JOB_NAME/ "
SCRIPT_ARGS+="--tensorboard_dir tensorboard_logs/$SLURM_JOB_NAME/ "

## RESUME
# SCRIPT_ARGS+="--resume_from_checkpoint checkpoints/$SLURM_JOB_NAME/$model_type-400steps "

# To use with conda directly
if [[ "$MODE" == "conda" ]]; then
    srun -l -D `pwd` conda run -p $CONDA_ENV_PATH --no-capture-output $SHELL_SCRIPT --hostfile $HOSTFILE $SCRIPT_ARGS
else
    # To use with smprun and docker
    # Make sure smprun is in path by adding SMModelParallelExamples/SMModelParallelExamples/bin to PATH
    smprun -v2 -i $CONTAINER_IMAGE \
    scripts/model.sh $@ $SCRIPT_ARGS
fi
