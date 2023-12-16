#!/usr/bin/env bash

# Sample cmd to start training:
#
# export HOSTS=`scontrol show hostnames | tr "\n" "," | sed 's/,$//g'`
# export MASTER_NODE=`scontrol show hostname | sort | head -1`
# export NUM_NODES=`scontrol show hostname | wc -l`
# export CONTAINER_NAME=rubik
# export WORK_DIR=$(dirname `pwd`)
# export SCRIPT=$WORK_DIR/main.py
#                                                                                            1           2            3            4            5
# srun --nodelist=$HOSTS docker exec $CONTAINER_NAME bash -c "cd $WORK_DIR; bash `pwd`/7b.sh $NUM_NODES  $MASTER_NODE $CONFIG_FILE $NSYS_OUTPUT $SCRIPT

### OLD version.
### num_nodes=${1:-1}
### master_node=${2:-`hostname`}
### config_file=${3}
### nsys_output=${4}
### script=${5:-"main.py"}
### shift 5

shift 1  # --hostfile
HOSTFILE=${1}
num_nodes=`cat $HOSTFILE| grep -v "^$" | wc -l`
master_node=`head -1 $HOSTFILE`

config_file=${2}
nsys_output=${3}
script=${4:-"main.py"}
shift 4

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


NSYS_CMD="nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --cuda-memory-usage=true --cudabacktrace=true -x true -o $nsys_output --force-overwrite=true"
CMD="--rdzv_endpoint=$MASTER_ADDR:29400 --rdzv_id=100 --rdzv_backend=c10d"
TORCH_CMD="torchrun --nnodes=${num_nodes} --nproc_per_node=8"

$NSYS_CMD \
$TORCH_CMD $CMD \
    $script \
    --config_file $config_file \
    $@
