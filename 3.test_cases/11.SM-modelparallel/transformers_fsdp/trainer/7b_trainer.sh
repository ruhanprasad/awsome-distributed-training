#!/usr/bin/env bash
#!/usr/bin/python

# In order to start training:
# srun --nodelist=$HOSTS docker exec <your_cont_name> bash -c "cd <folder>; bash run_fsdp.sh 2 compute-7"

num_nodes=${1:-1}
# master_node=$2
master_node=${2:-`hostname`}

export MASTER_ADDR=$master_node
export GPU_NUM_DEVICES=8
export NCCL_SOCKET_IFNAME="^lo,docker"
export RDMAV_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_DEBUG_SUBSYS=off
export NCCL_DEBUG="INFO"

export CUDA_LAUNCH_BLOCKING=0
export OFFLOAD_ACTIVATIONS=0
export OFFLOAD_IMPL="sagemaker"
export HIDDEN_WIDTH=4096
export NUM_LAYERS=32
export NUM_HEADS=32

CMD="--rdzv_endpoint=$MASTER_ADDR:29400 --rdzv_id=100 --rdzv_backend=c10d"
NSYS_CMD="nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --cuda-memory-usage=true --cudabacktrace=true -x true -o /fsx/users/huilgolr/nsys_profiles/fsdp_hybridzero2_7b_16node.out --force-overwrite=true"
if (($SLURM_PROCID == 0)); then
        # TORCH_CMD="$NSYS_CMD torchrun --nnodes=${num_nodes} --nproc_per_node=8"
        TORCH_CMD="torchrun --nnodes=${num_nodes} --nproc_per_node=8"
else
        TORCH_CMD="torchrun --nnodes=${num_nodes} --nproc_per_node=8"
fi

# TORCH_CMD="$NSYS_CMD torchrun --nnodes=${num_nodes} --nproc_per_node=8"
$TORCH_CMD $CMD trainer.py \
        --output_dir /fsx/user/huilgolr/checkpoints \
        --overwrite_output_dir \
        --do_train \
        --model_type gpt_neox \
        --tokenizer_name "EleutherAI/gpt-neox-20b" \
        --optim adamw_torch \
        --dataset_name wikitext \
        --dataset_config_name wikitext-2-raw-v1 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 4 \
        --dataloader_drop_last True \
        --learning_rate 0.0002 \
        --fsdp "full_shard auto_wrap" \
        --bf16 \
        --fsdp_config trainer_fsdp_config.json \
        --config_overrides "hidden_size=4096,num_attention_heads=32,num_hidden_layers=32,intermediate_size=16384"
