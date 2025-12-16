# Distributed Training Setup

This guide explains how to set up distributed AlphaZero training across two machines:

- **CPU Machine**: Powerful CPU for MCTS self-play (runs Ray head node)
- **GPU Machine**: RTX 3070 for neural network inference and training

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CPU MACHINE (Head Node)                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   Ray Head + Driver                              │   │
│  │  - Training orchestration                                        │   │
│  │  - Replay buffer actor                                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │ SelfPlay    │ │ SelfPlay    │ │ SelfPlay    │ │ SelfPlay    │ ...   │
│  │ Worker 1    │ │ Worker 2    │ │ Worker 3    │ │ Worker N    │       │
│  │ (MCTS+CPU)  │ │ (MCTS+CPU)  │ │ (MCTS+CPU)  │ │ (MCTS+CPU)  │       │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘       │
│         └───────────────┴───────┬───────┴───────────────┘              │
│                                 │ Inference requests                    │
└─────────────────────────────────┼───────────────────────────────────────┘
                                  │ LAN
┌─────────────────────────────────┼───────────────────────────────────────┐
│                    GPU MACHINE (Worker Node)                            │
│                                 ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                 Inference Server Actor (GPU)                      │   │
│  │  - Batched neural network inference                              │   │
│  │  - Returns (policy, value) predictions                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   Training Actor (GPU)                           │   │
│  │  - Gradient updates with mixed precision                         │   │
│  │  - Broadcasts updated weights                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

Both machines need:

- Python 3.12+
- `uv` package manager installed
- Network connectivity (same LAN recommended)
- Open firewall ports: 6379 (Ray GCS), 8265 (Dashboard), 10001-10100 (object transfer)

## Setup Instructions

### Step 1: Clone/Copy Project to Both Machines

Ensure both machines have the same codebase at the same path, or adjust PYTHONPATH accordingly.

```bash
# On both machines
cd /path/to/project
git clone <repo-url> Game
cd Game
```

### Step 2: Install Dependencies

**On CPU Machine (Head):**

```bash
cd /path/to/Game
uv sync --extra distributed
```

**On GPU Machine (Worker):**

```bash
cd /path/to/Game
uv sync --extra distributed

# Verify CUDA is available
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 3: Configure Constants (Optional)

Edit `constants.py` to adjust settings:

```python
# Ray Distributed Training Settings
RAY_LOCAL_MODE = False          # Set True to test locally without cluster
RAY_HEAD_ADDRESS = "auto"       # Will auto-connect when using scripts
RAY_INFERENCE_BATCH_SIZE = 32   # Batch size for GPU inference
RAY_NUM_WORKERS = 12            # Self-play workers on CPU machine
RAY_WEIGHT_SYNC_INTERVAL = 2    # Sync weights every N games
```

### Step 4: Start Ray Cluster

**On CPU Machine (Head Node):**

```bash
cd /path/to/Game
./scripts/start_head.sh
```

This will output the head address like:

```
Head node address: 192.168.1.100:6379
Dashboard URL: http://192.168.1.100:8265
```

**On GPU Machine (Worker Node):**

```bash
cd /path/to/Game
RAY_HEAD_IP=192.168.1.100 ./scripts/start_worker.sh
```

Replace `192.168.1.100` with your CPU machine's actual IP address.

### Step 5: Verify Cluster

Open the Ray Dashboard at `http://<cpu-machine-ip>:8265` to verify both nodes are connected.

Or run on the CPU machine:

```bash
ray status
```

You should see resources from both machines, including GPUs from the worker.

### Step 6: Start Training

**On CPU Machine:**

```bash
cd /path/to/Game
./scripts/run_distributed.sh
```

Or manually:

```bash
uv run python -m AI.distributed
```

## Local Testing Mode

To test without a cluster (runs everything on one machine):

```bash
# Edit constants.py
RAY_LOCAL_MODE = True

# Then run
uv run python -m AI.distributed
```

## Troubleshooting

### "No module named 'ray'" in workers

Ensure both machines have Ray installed:

```bash
uv sync --extra distributed
```

### Connection refused to head node

1. Check firewall allows port 6379
2. Verify head node IP is correct
3. Check Ray is running: `ray status`

### GPU not detected

```bash
# On GPU machine
uv run python -c "import torch; print(torch.cuda.is_available())"

# Check Ray sees the GPU
ray status | grep GPU
```

### Slow inference

- Increase `RAY_INFERENCE_BATCH_SIZE` (32-64)
- Check network latency between machines
- Ensure GPU machine is actually using CUDA

### Out of memory

- Reduce `RAY_NUM_WORKERS`
- Reduce `REPLAY_BUFFER_SIZE`
- Reduce `BATCH_SIZE`

## Stopping the Cluster

Run on both machines:

```bash
./scripts/stop_cluster.sh
# or
ray stop
```

## Performance Tips

1. **Network**: Use wired Ethernet connection between machines
2. **Batch size**: Larger `RAY_INFERENCE_BATCH_SIZE` = better GPU utilization
3. **Workers**: Set `RAY_NUM_WORKERS` to number of CPU cores - 2
4. **Checkpoints**: Checkpoints are saved on the head node (CPU machine)
