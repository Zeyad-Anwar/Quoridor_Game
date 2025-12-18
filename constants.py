# --- CONFIGURATION ---
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 1450
BG_COLOR = (25, 20, 15)  # Darker, richer brown

# Grid Settings
MARGIN_X = 50
MARGIN_Y = 50
TILE_SIZE = 130  # The size of the image on screen
GAP_SIZE = 15  # The space between tiles (for walls)
GRID_COUNT = 9

# Colors - Enhanced palette
PLAYER1_COLOR = (231, 76, 60)  # Modern red
PLAYER1_GLOW = (255, 120, 100)  # Light red glow
PLAYER2_COLOR = (52, 152, 219)  # Modern blue
PLAYER2_GLOW = (100, 180, 255)  # Light blue glow
WALL_COLOR = (160, 100, 50)  # Warmer brown wood color
WALL_HIGHLIGHT = (200, 140, 80)  # Wall highlight
WALL_PREVIEW_COLOR = (180, 160, 120)  # Light tan for preview
WALL_INVALID_COLOR = (220, 60, 60)  # Red for invalid placement
HIGHLIGHT_COLOR = (46, 204, 113)  # Modern green for valid moves
HIGHLIGHT_PULSE = (100, 230, 150)  # Brighter green for pulse effect

# UI Colors - Modern dark theme
BUTTON_COLOR = (50, 45, 40)  # Dark charcoal
BUTTON_HOVER_COLOR = (70, 65, 55)  # Lighter on hover
BUTTON_BORDER = (120, 100, 80)  # Warm border
BUTTON_GLOW = (180, 150, 100)  # Golden glow on hover
TEXT_COLOR = (245, 245, 245)  # Off-white
TEXT_SECONDARY = (180, 175, 165)  # Muted text
ACCENT_COLOR = (218, 165, 32)  # Golden accent
PANEL_COLOR = (35, 30, 25)  # Panel background
PANEL_BORDER = (80, 70, 55)  # Panel border

# Game Settings
PAWN_RADIUS = 40
WALL_THICKNESS = 10
FPS = 60

# Game Modes
MODE_MENU = "menu"
MODE_PVP = "pvp"
MODE_PVE = "pve"
MODE_LOAD = "load"

# Save/Load Settings
SAVES_DIR = "saves"

# --- GAMEPLAY (Human vs AI) ---
# Which agent to use for Player-vs-AI in the pygame UI.
# - "mcts": AlphaZero MCTS + neural network
# - "alphabeta": alpha-beta + heuristic (no NN required)
PVE_AI_AGENT = "alphabeta"

# Alpha-beta difficulty mapping for gameplay.
PVE_ALPHABETA_DEPTH_EASY = 1
PVE_ALPHABETA_DEPTH_MEDIUM = 2
PVE_ALPHABETA_DEPTH_HARD = 3

# AI Settings (Pure MCTS - Legacy)
MCTS_TIME_LIMIT = 2.0  # Seconds for AI to think
MCTS_SIMULATIONS = 400  # Max simulations per move

# AlphaZero MCTS Settings
ALPHA_MCTS_SIMULATIONS = 2500  # Increased for better policy quality
C_PUCT = 1.5  # Exploration constant for PUCT formula
DIRICHLET_ALPHA = 0.1  # Reduced: ~10/num_actions for proper exploration noise
DIRICHLET_EPSILON = 0.25  # Weight of Dirichlet noise at root

# Action Space
# Moves: 81 (any position on 9x9 board)
# Walls: 128 (64 horizontal + 64 vertical, each on 8x8 grid)
ACTION_SPACE_SIZE = 209

# Neural Network Architecture
NUM_RES_BLOCKS = 8  # Number of residual blocks (increased for stronger play)
NUM_FILTERS = 128  # Filters in convolutional layers
INPUT_CHANNELS = 8  # Input planes: 2 positions + 2 walls + 3 meta + 1 move number

# Training Hyperparameters
SELF_PLAY_GAMES = 50  # Increased for more training data per iteration
REPLAY_BUFFER_SIZE = 100_000  # Max training examples to store
BATCH_SIZE = 512  # Increased for better gradient estimates
LEARNING_RATE = 0.0005  # Reduced from 0.002 to prevent overshooting
WEIGHT_DECAY = 1e-4
TRAINING_ITERATIONS = 100  # Total training iterations
CHECKPOINT_INTERVAL = 10  # Save model every N iterations
NUM_PARALLEL_GAMES = 18  # Parallel self-play games
AUTO_SAVE_MINUTES = 10  # Auto-save checkpoint every N minutes
EVAL_INTERVAL = 5  # Evaluate against random every N iterations
EVAL_GAMES = 20  # Number of evaluation games (increased for reliable measurement)

# MCTS Batch Size - controls how many leaf nodes are evaluated at once
# Higher = better GPU utilization but more memory usage
MCTS_BATCH_SIZE = 1024  # Batch size for neural network inference in MCTS

# Self-play Settings
TEMP_THRESHOLD = 20  # Use temp=1 for first N moves, then temp→0
TEMP_INIT = 1.0  # Initial temperature
TEMP_FINAL = 0.05  # Final temperature after threshold (sharper play)

# Pure Self-Play (No Bootstrap)
# AlphaZero uses self-play from iteration 1 with randomly initialized network

# Difficulty Presets for Gameplay
# Each preset defines MCTS simulations and temperature for AI player
DIFFICULTY_PRESETS = {
    "easy": {
        "num_simulations": 50,
        "temperature": 0.8,
        "description": "Casual play with some randomness",
    },
    "medium": {
        "num_simulations": 500,
        "temperature": 0.4,
        "description": "Balanced challenge",
    },
    "hard": {
        "num_simulations": 2500,
        "temperature": 0,
        "description": "Near-optimal play",
    },
}

# --- PRIORITY EXPERIENCE REPLAY (PER) ---
# Prioritization exponent (0 = uniform sampling, 1 = full priority)
PER_ALPHA = 0.6

# Initial importance sampling correction (annealed to 1.0)
PER_BETA_START = 0.4

# Number of training iterations to anneal beta to 1.0
PER_BETA_ITERATIONS = TRAINING_ITERATIONS

# Small constant to prevent zero priority
PER_EPSILON = 1e-6

# --- EXPONENTIAL MOVING AVERAGE (EMA) TARGET NETWORK ---
# EMA decay rate for target network (higher = slower update)
EMA_DECAY = 0.995

# Update EMA network every N training batches
EMA_UPDATE_FREQ = 1

# --- PROGRESSIVE WINDOW FOR SELF-PLAY DATA ---
# Maximum age (in iterations) for examples to receive full weight
PROGRESSIVE_WINDOW_SIZE = 20

# Weight factor for recency bias (0 = uniform, 1 = only recent)
PROGRESSIVE_RECENCY_WEIGHT = 0.7

# --- RAY DISTRIBUTED TRAINING ---
# Set to True to run everything locally in a single process (for debugging)
RAY_LOCAL_MODE = False

# Set to True to force all actors to run on CPU (useful for testing on CPU-only machine)
# When False, GPU actors will wait for a GPU worker to connect
RAY_CPU_ONLY_MODE = False

# Head node address - set to the IP of the CPU machine running ray start --head
# Format: "ip:port" or "auto" to connect to existing cluster
RAY_HEAD_ADDRESS = "auto"

# Port for Ray head node (used when starting cluster)
RAY_HEAD_PORT = 6379

# Batch size for remote inference requests (higher = better throughput, more latency)
RAY_INFERENCE_BATCH_SIZE = 1024

# How often workers sync weights from the training actor (in games)
RAY_WEIGHT_SYNC_INTERVAL = 2

# Number of self-play workers to spawn on CPU node
RAY_NUM_WORKERS = 12

# Object store memory for Ray (in bytes, None = auto)
RAY_OBJECT_STORE_MEMORY = None

# --- SELF-PLAY AGENT SELECTION ---
# Controls how self-play selects actions.
# - "mcts": AlphaZero MCTS + neural network (default)
# - "alphabeta": alpha-beta pruning + heuristic (bootstrap/baseline)
SELFPLAY_AGENT = "mcts"

# Optional bootstrap: use a different agent for the first N training iterations,
# then fall back to SELFPLAY_AGENT.
BOOTSTRAP_SELFPLAY_AGENT = "alphabeta"
BOOTSTRAP_SELFPLAY_ITERS = 0

# Alpha-beta settings (used when agent is "alphabeta")
ALPHABETA_DEPTH = 2
ALPHABETA_MAX_MOVES_PER_POSITION = 32
ALPHABETA_MAX_WALL_CANDIDATES = 24
ALPHABETA_WALL_NEIGHBORHOOD = 2
