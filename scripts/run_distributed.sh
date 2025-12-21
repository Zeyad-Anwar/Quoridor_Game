#!/bin/bash
# Run distributed training
# Execute this on the HEAD node (CPU machine) after both nodes are connected

set -e

# Change to project directory
cd "$(dirname "$0")/.."

echo "=============================================="
echo "Starting Distributed Training"
echo "=============================================="
echo ""

# Check Ray cluster status
echo "Checking Ray cluster status..."
if ! ray status 2>/dev/null; then
    echo ""
    echo "ERROR: Ray cluster is not running!"
    echo ""
    echo "Please start the cluster first:"
    echo "  1. On this machine (CPU): ./scripts/start_head.sh"
    echo "  2. On GPU machine: RAY_HEAD_IP=<this-ip> ./scripts/start_worker.sh"
    echo ""
    exit 1
fi

echo ""
ray status
echo ""

# Check for GPU resources
GPU_COUNT=$(ray status 2>/dev/null | grep -oP 'GPU: \K[0-9.]+' | head -1 || echo "0")

if [ "$GPU_COUNT" = "0" ] || [ -z "$GPU_COUNT" ]; then
    echo ""
    echo "WARNING: No GPU resources detected in cluster!"
    echo ""
    echo "The GPU worker might not be connected yet."
    echo "Training will still work but inference will be slow."
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run training
echo ""
echo "Starting distributed training..."
echo ""

# Use uv to run the training script
uv run python -m AI.distributed "$@"
