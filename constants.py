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
MCTS_SIMULATIONS = 1000  # Max simulations per move

# AlphaZero MCTS Settings
ALPHA_MCTS_SIMULATIONS = 800      # Simulations per move
C_PUCT = 1.5                       # Exploration constant for PUCT formula
DIRICHLET_ALPHA = 0.3              # Noise parameter for root exploration
DIRICHLET_EPSILON = 0.25           # Weight of Dirichlet noise at root

# Action Space
# Moves: 81 (any position on 9x9 board)
# Walls: 128 (64 horizontal + 64 vertical, each on 8x8 grid)
ACTION_SPACE_SIZE = 209

# Neural Network Architecture
NUM_RES_BLOCKS = 5                 # Number of residual blocks
NUM_FILTERS = 128                  # Filters in convolutional layers
INPUT_CHANNELS = 7                 # Input planes: 2 positions + 2 walls + 3 meta

# Training Hyperparameters
SELF_PLAY_GAMES = 100              # Games per training iteration
REPLAY_BUFFER_SIZE = 50000         # Max training examples to store
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
TRAINING_ITERATIONS = 100
CHECKPOINT_INTERVAL = 10           # Save model every N iterations

# Self-play Settings
TEMP_THRESHOLD = 15                # Use temp=1 for first N moves, then temp→0
TEMP_INIT = 1.0                    # Initial temperature
TEMP_FINAL = 0.1                   # Final temperature after threshold