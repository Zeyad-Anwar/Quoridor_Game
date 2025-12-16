#!/bin/bash
# Start Ray head node on CPU machine
# Run this script on the machine with the powerful CPU

set -e

# Configuration
HEAD_PORT=${RAY_HEAD_PORT:-6379}
DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
NUM_CPUS=${RAY_NUM_CPUS:-}  # Empty = auto-detect
OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-}  # Empty = auto

echo "=============================================="
echo "Starting Ray Head Node (CPU Machine)"
echo "=============================================="
echo ""

# Check if Ray is already running
if ray status 2>/dev/null; then
    echo "Ray is already running. Stopping existing cluster..."
    ray stop --force
    sleep 2
fi

# Build the ray start command
RAY_CMD="ray start --head --port=$HEAD_PORT --dashboard-port=$DASHBOARD_PORT"

if [ -n "$NUM_CPUS" ]; then
    RAY_CMD="$RAY_CMD --num-cpus=$NUM_CPUS"
fi

if [ -n "$OBJECT_STORE_MEMORY" ]; then
    RAY_CMD="$RAY_CMD --object-store-memory=$OBJECT_STORE_MEMORY"
fi

# Start Ray head
echo "Running: $RAY_CMD"
$RAY_CMD

# Get the IP address
HEAD_IP=$(hostname -I | awk '{print $1}')

echo ""
echo "=============================================="
echo "Ray Head Node Started!"
echo "=============================================="
echo ""
echo "Head node address: $HEAD_IP:$HEAD_PORT"
echo "Dashboard URL: http://$HEAD_IP:$DASHBOARD_PORT"
echo ""
echo "To connect the GPU worker, run on the GPU machine:"
echo ""
echo "  ray start --address='$HEAD_IP:$HEAD_PORT' --num-gpus=1"
echo ""
echo "Or use the start_worker.sh script with:"
echo ""
echo "  RAY_HEAD_IP=$HEAD_IP ./scripts/start_worker.sh"
echo ""
echo "=============================================="
echo ""
echo "To stop the cluster, run: ray stop"
echo ""
