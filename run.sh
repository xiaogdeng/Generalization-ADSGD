#!/bin/bash

MASTER_ADDR=`scontrol show hostname $SLURM_NODELIST| head -n 1`
MASTER_PORT=22345
GPUS_PER_NODE=4

NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID

export NCCL_SOCKET_IFNAME=ib0

python -m torch.distributed.launch --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES \
                                   --node_rank $NODE_RANK --master_addr $MASTER_ADDR \
                                   --master_port $MASTER_PORT \
                                   AD-SGD.py --lr=$1 --delay_type=$2 --scheduler=$3 \
                                   --batchsize=$4 --topo=$5 --model=$6 --dataset=$7 --seed=$8 --epoch=$9
                                   
                                   
