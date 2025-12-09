# --- CONFIGURATION ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
BG_COLOR = (45, 30, 15)  # Dark Brown 

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

# AI Settings
MCTS_TIME_LIMIT = 2.0  # Seconds for AI to think
MCTS_SIMULATIONS = 400  # Max simulations per move
