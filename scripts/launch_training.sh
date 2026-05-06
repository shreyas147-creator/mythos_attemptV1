#!/bin/bash
# Launch script for Mythos Transformer training
# Supports single-node and multi-node distributed training with DeepSpeed

set -e

# Default values
CONFIG_FILE=${1:-"configs/mythos_10t_full.yaml"}
NUM_GPUS=${NUM_GPUS:-8}
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29500}

echo "=========================================="
echo "Mythos Transformer Training Launch"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "GPUs per node: $NUM_GPUS"
echo "Number of nodes: $NUM_NODES"
echo "Node rank: $NODE_RANK"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "=========================================="

# Calculate total GPUs
TOTAL_GPUS=$((NUM_GPUS * NUM_NODES))
echo "Total GPUs: $TOTAL_GPUS"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_SOCKET_IFNAME=eth0
export PYTHONUNBUFFERED=1

# DeepSpeed configuration
DS_CONFIG="configs/deepspeed_config.json"

# Create DeepSpeed config from YAML
python scripts/generate_ds_config.py --config $CONFIG_FILE --output $DS_CONFIG

# Launch training with DeepSpeed
if [ $NUM_NODES -eq 1 ]; then
    # Single-node training
    echo "Launching single-node training..."
    deepspeed --num_gpus=$NUM_GPUS \
              src/training/trainer.py \
              --config $CONFIG_FILE
else
    # Multi-node training
    echo "Launching multi-node training..."
    deepspeed --num_gpus=$NUM_GPUS \
              --num_nodes=$NUM_NODES \
              --node_rank=$NODE_RANK \
              --master_addr=$MASTER_ADDR \
              --master_port=$MASTER_PORT \
              src/training/trainer.py \
              --config $CONFIG_FILE
fi

echo "=========================================="
echo "Training completed!"
echo "=========================================="
