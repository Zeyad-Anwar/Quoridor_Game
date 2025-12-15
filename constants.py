# --- CONFIGURATION ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
BG_COLOR = (45, 30, 15)  # Dark Brown (This acts as the color of the GROOVES)

# Grid Settings
MARGIN_X = 50       
MARGIN_Y = 50       
TILE_SIZE = 65      # The size of the image on screen
GAP_SIZE = 15       # The space between tiles (for walls)
GRID_COUNT = 9

# Colors
PLAYER1_COLOR = (220, 50, 50)    # Red
PLAYER2_COLOR = (50, 50, 220)    # Blue
WALL_COLOR = (139, 90, 43)       # Brown wood color
WALL_PREVIEW_COLOR = (200, 180, 140)  # Light tan for preview
WALL_INVALID_COLOR = (200, 50, 50)    # Red for invalid placement
HIGHLIGHT_COLOR = (0, 255, 0)    # Green for valid moves
BUTTON_COLOR = (70, 50, 30)      # Dark brown for buttons
BUTTON_HOVER_COLOR = (100, 70, 45)
TEXT_COLOR = (255, 255, 255)     # White

# Game Settings
PAWN_RADIUS = 25
WALL_THICKNESS = 10
FPS = 60

# Game Modes
MODE_MENU = 'menu'
MODE_PVP = 'pvp'
MODE_PVE = 'pve'

# AI Settings (Pure MCTS - Legacy)
MCTS_TIME_LIMIT = 2.0  # Seconds for AI to think
MCTS_SIMULATIONS = 250  # Max simulations per move

# AlphaZero MCTS Settings
ALPHA_MCTS_SIMULATIONS = 400       # Increased for better policy quality
C_PUCT = 1.5                       # Exploration constant for PUCT formula
DIRICHLET_ALPHA = 0.1              # Reduced: ~10/num_actions for proper exploration noise
DIRICHLET_EPSILON = 0.25           # Weight of Dirichlet noise at root

# Action Space
# Moves: 81 (any position on 9x9 board)
# Walls: 128 (64 horizontal + 64 vertical, each on 8x8 grid)
ACTION_SPACE_SIZE = 209

# Neural Network Architecture
NUM_RES_BLOCKS = 8                 # Number of residual blocks (increased for stronger play)
NUM_FILTERS = 128                  # Filters in convolutional layers
INPUT_CHANNELS = 8                 # Input planes: 2 positions + 2 walls + 3 meta + 1 move number

# Training Hyperparameters
SELF_PLAY_GAMES = 50               # Increased for more training data per iteration
REPLAY_BUFFER_SIZE = 100_000       # Max training examples to store
BATCH_SIZE = 512                   # Increased for better gradient estimates
LEARNING_RATE = 0.001              # Reduced from 0.002 to prevent overshooting
WEIGHT_DECAY = 1e-4
TRAINING_ITERATIONS = 50          # Total training iterations
CHECKPOINT_INTERVAL = 20           # Save model every N iterations
NUM_PARALLEL_GAMES = 14             # Parallel self-play games
AUTO_SAVE_MINUTES = 10             # Auto-save checkpoint every N minutes
EVAL_INTERVAL = 10                 # Evaluate against random every N iterations
EVAL_GAMES = 14                    # Number of evaluation games (increased for reliable measurement)

# MCTS Batch Size - controls how many leaf nodes are evaluated at once
# Higher = better GPU utilization but more memory usage
MCTS_BATCH_SIZE = 16                # Batch size for neural network inference in MCTS

# Self-play Settings
TEMP_THRESHOLD = 20                # Use temp=1 for first N moves, then temp→0
TEMP_INIT = 1.0                    # Initial temperature
TEMP_FINAL = 0.05                  # Final temperature after threshold (sharper play)

# Curriculum Learning Settings
WIN_RATE_THRESHOLD = 0.6           # Win rate vs random to switch to self-play (reduced from 0.8 to shorten bootstrap)
BOOTSTRAP_TEMP_INIT = 1.0          # Initial temperature during bootstrap
BOOTSTRAP_TEMP_FINAL = 0.05        # Final temperature during bootstrap
BOOTSTRAP_TEMP_THRESHOLD = 20      # Move number to drop temperature

# Difficulty Presets for Gameplay
# Each preset defines MCTS simulations and temperature for AI player
DIFFICULTY_PRESETS = {
    'easy': {
        'num_simulations': 50,
        'temperature': 0.8,
        'description': 'Casual play with some randomness'
    },
    'medium': {
        'num_simulations': 500,
        'temperature': 0.4,
        'description': 'Balanced challenge'
    },
    'hard': {
        'num_simulations': 1000,
        'temperature': 0,
        'description': 'Near-optimal play'
    }
}