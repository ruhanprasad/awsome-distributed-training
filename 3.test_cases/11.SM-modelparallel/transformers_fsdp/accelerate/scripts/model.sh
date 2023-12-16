#!/usr/bin/env bash

parse_inputs() {
    while [[ $# -gt 0 ]]; do
        key="$1"
        case $key in
        --hostfile)
            hostfile=$2
            shift 2
            ;;
        *)
            shift 1
            ;;
        esac
    done
}

parse_inputs $@

if [ -z "$hostfile" ]; then
    echo "Hostfile needs to be passed"
    exit 1
fi

num_nodes=$(cat $hostfile | wc -l)

model_name="/fsx/users/viczhu/third_party/huggingface/NousResearch/Llama-2-7b-hf"
transformer_model_type="llama_v2"

export SM_NUM_GPUS=8

export MASTER_ADDR=$(head -n 1 $hostfile)
export GPU_NUM_DEVICES=8
export NCCL_PROTO="simple"
export NCCL_SOCKET_IFNAME="^lo,docker"
export RDMAV_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_DEBUG_SUBSYS=off
export NCCL_DEBUG="INFO"

CMD="--rdzv_endpoint=$MASTER_ADDR:29400 --rdzv_id=100 --rdzv_backend=c10d"
TORCH_CMD="torchrun --nnodes=${num_nodes} --nproc_per_node=8"

$TORCH_CMD $CMD \
    run_clm_no_trainer.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path ${model_name} \
    --transformer_model_type ${transformer_model_type} \
    --activation_checkpointing \
    --auto_wrap_policy transformer_auto_wrap_policy \
    --sharding_strategy full_shard \
    --mixed_precision_dtype bf16 \
    --max_train_steps 100 \
    --checkpointing_steps 50 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --block_size 512 \
    --trust_remote_code True \
    --output_dir tmp/test-clm