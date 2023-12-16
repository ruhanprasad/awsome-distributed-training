#!/usr/bin/env bash

# Sample cmd to start training:
#
# export HOSTS=`scontrol show hostnames | tr "\n" "," | sed 's/,$//g'`
# export MASTER_NODE=`scontrol show hostname | sort | head -1`
# export NUM_NODES=`scontrol show hostname | wc -l`
# export CONTAINER_NAME=rubik
# export WORK_DIR=$(dirname `pwd`)
# export SCRIPT=$WORK_DIR/main.py
#                                                                                            1           2            3            4
# srun --nodelist=$HOSTS docker exec $CONTAINER_NAME bash -c "cd $WORK_DIR; bash `pwd`/7b.sh $NUM_NODES  $MASTER_NODE $CONFIG_FILE $SCRIPT

### OLD version.
### num_nodes=${1:-1}
### master_node=${2:-`hostname`}
### config_file=${3}
### script=${4:-"main.py"}
### shift 4

shift 1  # --hostfile
HOSTFILE=${1}
num_nodes=`cat $HOSTFILE| grep -v "^$" | wc -l`
master_node=`head -1 $HOSTFILE`

config_file=${2}
script=${3:-"main.py"}
shift 3

export SM_NUM_GPUS=8
export SM_CHANNEL_TRAIN="/fsx/skodgule/data/train_ids_wsvocab_redo_2048_smaller/"
export SM_CHANNEL_TEST="/fsx/skodgule/data/val_ids_wsvocab_2048/"

export MASTER_ADDR=$master_node
export GPU_NUM_DEVICES=8
export NCCL_PROTO="simple"
export NCCL_SOCKET_IFNAME="^lo,docker"
export RDMAV_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_DEBUG_SUBSYS=off
export NCCL_DEBUG="INFO"


CMD="--rdzv_endpoint=$MASTER_ADDR:29400 --rdzv_id=100 --rdzv_backend=c10d"
TORCH_CMD="torchrun --nnodes=${num_nodes} --nproc_per_node=8"
# TORCH_CMD="$NSYS_CMD torchrun --nnodes=${num_nodes} --nproc_per_node=8"

$TORCH_CMD $CMD \
    $script \
    --config_file $config_file \
    $@
