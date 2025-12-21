#!/bin/bash
# Start Ray worker node on GPU machine
# Run this script on the machine with RTX 3070

set -e

# Configuration - SET THIS TO YOUR HEAD NODE IP!
HEAD_IP=${RAY_HEAD_IP:-"192.168.0.243"}
HEAD_PORT=${RAY_HEAD_PORT:-6379}
NUM_GPUS=${RAY_NUM_GPUS:-1}
NUM_CPUS=${RAY_NUM_CPUS:-}  # Empty = auto-detect

echo "=============================================="
echo "Starting Ray Worker Node (GPU Machine)"
echo "=============================================="
echo ""

# Validate HEAD_IP
if [ "$HEAD_IP" = "192.168.0.243" ]; then
    echo "ERROR: Please set RAY_HEAD_IP environment variable"
    echo ""
    echo "Usage:"
    echo "  RAY_HEAD_IP=192.168.1.100 ./scripts/start_worker.sh"
    echo ""
    echo "Or export it first:"
    echo "  export RAY_HEAD_IP=192.168.1.100"
    echo "  ./scripts/start_worker.sh"
    echo ""
    exit 1
fi

# Check if Ray is already running
if ray status 2>/dev/null; then
    echo "Ray is already running. Stopping existing instance..."
    ray stop --force
    sleep 2
fi

# Build the ray start command
RAY_CMD="ray start --address='$HEAD_IP:$HEAD_PORT' --num-gpus=$NUM_GPUS"

if [ -n "$NUM_CPUS" ]; then
    RAY_CMD="$RAY_CMD --num-cpus=$NUM_CPUS"
fi

# Start Ray worker
echo "Connecting to head node at $HEAD_IP:$HEAD_PORT"
echo "Running: $RAY_CMD"
eval $RAY_CMD

echo ""
echo "=============================================="
echo "Ray Worker Node Started!"
echo "=============================================="
echo ""
echo "Connected to head: $HEAD_IP:$HEAD_PORT"
echo "GPUs available: $NUM_GPUS"
echo ""
echo "Dashboard URL: http://$HEAD_IP:8265"
echo ""
echo "=============================================="
echo ""
echo "To stop this worker, run: ray stop"
echo ""
