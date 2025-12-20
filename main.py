"""
Quoridor Game - Main Entry Point
A strategic board game with pygame implementation.
"""
import os
import pygame
import sys

from constants import *
from game import GameState, Wall, Position, save_game, load_game, get_save_files
from AI.alpha_beta import AlphaBetaConfig, AlphaBetaPlayer


# --- LAYOUT MANAGER ---

class LayoutManager:
    """Manages dynamic layout calculations based on window size."""
    
    def __init__(self, screen_width: int, screen_height: int):
        self.update(screen_width, screen_height)
    
    def update(self, screen_width: int, screen_height: int) -> None:
        """Recalculate all layout values based on new screen dimensions."""
        # Enforce minimum window size
        self.screen_width = max(screen_width, MIN_SCREEN_WIDTH)
        self.screen_height = max(screen_height, MIN_SCREEN_HEIGHT)
        
        # Calculate the available space for the board
        # Reserve space for UI elements (top: 60px, bottom: 120px, sides: 20px each)
        ui_top_margin = 60
        ui_bottom_margin = 120
        ui_side_margin = 20
        
        available_width = self.screen_width - (ui_side_margin * 2)
        available_height = self.screen_height - ui_top_margin - ui_bottom_margin
        
        # The board is square, so use the smaller dimension
        available_board_size = min(available_width, available_height)
        
        # Calculate tile size based on available space
        # Board = GRID_COUNT * TILE_SIZE + (GRID_COUNT - 1) * GAP_SIZE
        # GAP_SIZE is typically ~11.5% of TILE_SIZE
        # So: Board ≈ GRID_COUNT * TILE_SIZE + (GRID_COUNT - 1) * 0.115 * TILE_SIZE
        # Board ≈ TILE_SIZE * (GRID_COUNT + (GRID_COUNT - 1) * 0.115)
        # Board ≈ TILE_SIZE * (9 + 8 * 0.115) = TILE_SIZE * 9.92
        
        gap_ratio = 0.115  # Gap is 11.5% of tile size
        total_units = GRID_COUNT + (GRID_COUNT - 1) * gap_ratio
        
        self.tile_size = int(available_board_size / total_units)
        self.gap_size = max(4, int(self.tile_size * gap_ratio))  # Minimum gap of 4px
        
        # Recalculate actual board size
        self.board_size = GRID_COUNT * self.tile_size + (GRID_COUNT - 1) * self.gap_size
        
        # Center the board horizontally, offset from top for UI
        self.margin_x = (self.screen_width - self.board_size) // 2
        self.margin_y = ui_top_margin + (available_height - self.board_size) // 2
        
        # Scale other game elements proportionally
        # Base reference: TILE_SIZE=130 -> PAWN_RADIUS=40, WALL_THICKNESS=10
        scale_factor = self.tile_size / 130.0
        
        self.pawn_radius = max(15, int(40 * scale_factor))
        self.wall_thickness = max(4, int(10 * scale_factor))
        
        # Font sizes based on screen height
        font_scale = self.screen_height / DEFAULT_SCREEN_HEIGHT
        self.title_font_size = max(48, int(96 * font_scale))
        self.large_font_size = max(36, int(72 * font_scale))
        self.font_size = max(18, int(36 * font_scale))
        self.small_font_size = max(14, int(28 * font_scale))
        self.tiny_font_size = max(12, int(22 * font_scale))
        
        # UI element sizes based on screen dimensions
        self.button_height = max(35, int(55 * font_scale))
        self.panel_padding = max(5, int(10 * font_scale))


# Global layout manager instance
layout = LayoutManager(DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT)

# Global fonts (will be recreated on resize)
title_font = None
large_font = None
font = None
small_font = None
tiny_font = None


def create_fonts():
    """Create fonts based on current layout sizes."""
    global title_font, large_font, font, small_font, tiny_font
    try:
        title_font = pygame.font.Font(None, layout.title_font_size)
        large_font = pygame.font.Font(None, layout.large_font_size)
        font = pygame.font.Font(None, layout.font_size)
        small_font = pygame.font.Font(None, layout.small_font_size)
        tiny_font = pygame.font.Font(None, layout.tiny_font_size)
    except:
        title_font = pygame.font.SysFont('arial', int(layout.title_font_size * 0.75))
        large_font = pygame.font.SysFont('arial', int(layout.large_font_size * 0.75))
        font = pygame.font.SysFont('arial', int(layout.font_size * 0.75))
        small_font = pygame.font.SysFont('arial', int(layout.small_font_size * 0.75))
        tiny_font = pygame.font.SysFont('arial', int(layout.tiny_font_size * 0.75))


# --- AI MODEL LOADING ---

CHECKPOINT_DIR = "checkpoints"

def get_latest_checkpoint() -> str | None:
    """Get the path to the latest checkpoint file."""
    if not os.path.exists(CHECKPOINT_DIR):
        return None
    
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')]
    if not checkpoints:
        return None
    
    # Prefer model_final.pt if it exists
    if 'model_autosave.pt' in checkpoints:
        return os.path.join(CHECKPOINT_DIR, 'model_autosave.pt')
    
    # Otherwise get the highest iteration number
    iter_checkpoints = [f for f in checkpoints if f.startswith('model_iter_')]
    if iter_checkpoints:
        iter_checkpoints.sort(key=lambda x: int(x.split('_')[-1].replace('.pt', '')))
        return os.path.join(CHECKPOINT_DIR, iter_checkpoints[-1])
    
    # Fallback to any checkpoint
    return os.path.join(CHECKPOINT_DIR, checkpoints[0])


def load_ai_player(player: int, difficulty: str = 'medium'):
    """Load the AI player for PVE.
    
    Args:
        player: Player number (1 or 2)
        difficulty: Difficulty level ('easy', 'medium', 'hard')
    """
    if difficulty not in DIFFICULTY_PRESETS:
        print(f"Unknown difficulty '{difficulty}', using 'medium'")
        difficulty = 'medium'

    # Allow swapping the gameplay opponent without touching the UI.
    if PVE_AI_AGENT == "alphabeta":
        depth = {
            "easy": PVE_ALPHABETA_DEPTH_EASY,
            "medium": PVE_ALPHABETA_DEPTH_MEDIUM,
            "hard": PVE_ALPHABETA_DEPTH_HARD,
        }[difficulty]
        print(f"Using AlphaBeta AI (depth={depth})")
        return AlphaBetaPlayer(
            player=player,
            cfg=AlphaBetaConfig(depth=depth),
            temperature=0.0,
        )

    # Default: MCTS + neural network
    settings = DIFFICULTY_PRESETS[difficulty]
    num_simulations = settings['num_simulations']
    temperature = settings['temperature']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    network = QuoridorNet()
    
    checkpoint_path = get_latest_checkpoint()
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading AI model from: {checkpoint_path}")
        print(f"Difficulty: {difficulty.upper()} (sims={num_simulations}, temp={temperature})")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    network.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    network.load_state_dict(checkpoint['state_dict'])
                else:
                    network.load_state_dict(checkpoint)
            else:
                network.load_state_dict(checkpoint)
        except RuntimeError as e:
            if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                print("Warning: Checkpoint architecture mismatch (old model has different NUM_RES_BLOCKS)")
                print("Using untrained network with current architecture.")
            else:
                raise
    else:
        print("Warning: No checkpoint found, using untrained network!")
    
    network.to(device)
    network.eval()
    
    return AlphaZeroPlayer(
        player=player, 
        network=network, 
        num_simulations=num_simulations,
        temperature=temperature
    )


# --- SETUP ---
pygame.init()
screen = pygame.display.set_mode(
    (DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT), 
    pygame.RESIZABLE
)
pygame.display.set_caption("Quoridor")
clock = pygame.time.Clock()

# Initialize fonts
create_fonts()

# Load Assets - Keep raw image for rescaling
raw_tile_img = pygame.image.load('assets/tile.png')
tile_img = pygame.transform.scale(raw_tile_img, (layout.tile_size, layout.tile_size))


def rescale_assets():
    """Rescale game assets based on current layout."""
    global tile_img
    tile_img = pygame.transform.scale(raw_tile_img, (layout.tile_size, layout.tile_size))


def handle_resize(new_width: int, new_height: int) -> pygame.Surface:
    """Handle window resize event and return the new screen surface."""
    global screen
    
    # Enforce minimum size
    new_width = max(new_width, MIN_SCREEN_WIDTH)
    new_height = max(new_height, MIN_SCREEN_HEIGHT)
    
    # Update layout calculations
    layout.update(new_width, new_height)
    
    # Recreate screen with new size
    screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)
    
    # Rescale assets and recreate fonts
    rescale_assets()
    create_fonts()
    
    return screen

# Animation state
animation_tick = 0


# --- UI HELPER FUNCTIONS ---

def draw_gradient_rect(surface: pygame.Surface, rect: pygame.Rect, 
                       color1: tuple, color2: tuple, vertical: bool = True) -> None:
    """Draw a rectangle with gradient fill."""
    if vertical:
        for i in range(rect.height):
            ratio = i / rect.height
            r = int(color1[0] + (color2[0] - color1[0]) * ratio)
            g = int(color1[1] + (color2[1] - color1[1]) * ratio)
            b = int(color1[2] + (color2[2] - color1[2]) * ratio)
            pygame.draw.line(surface, (r, g, b), 
                           (rect.x, rect.y + i), (rect.x + rect.width, rect.y + i))
    else:
        for i in range(rect.width):
            ratio = i / rect.width
            r = int(color1[0] + (color2[0] - color1[0]) * ratio)
            g = int(color1[1] + (color2[1] - color1[1]) * ratio)
            b = int(color1[2] + (color2[2] - color1[2]) * ratio)
            pygame.draw.line(surface, (r, g, b),
                           (rect.x + i, rect.y), (rect.x + i, rect.y + rect.height))


def draw_panel(rect: pygame.Rect, alpha: int = 220) -> None:
    """Draw a semi-transparent panel with border."""
    panel = pygame.Surface((rect.width, rect.height))
    panel.set_alpha(alpha)
    panel.fill(PANEL_COLOR)
    screen.blit(panel, rect.topleft)
    pygame.draw.rect(screen, PANEL_BORDER, rect, 2, border_radius=10)


def draw_shadow(rect: pygame.Rect, offset: int = 4, alpha: int = 60) -> None:
    """Draw a shadow behind a rectangle."""
    shadow = pygame.Surface((rect.width, rect.height))
    shadow.set_alpha(alpha)
    shadow.fill((0, 0, 0))
    screen.blit(shadow, (rect.x + offset, rect.y + offset))


# --- HELPER FUNCTIONS ---

def get_tile_center(row: int, col: int) -> tuple[int, int]:
    """Get the pixel center of a tile at (row, col)."""
    x = layout.margin_x + (col * (layout.tile_size + layout.gap_size)) + layout.tile_size // 2
    y = layout.margin_y + (row * (layout.tile_size + layout.gap_size)) + layout.tile_size // 2
    return (x, y)


def get_tile_topleft(row: int, col: int) -> tuple[int, int]:
    """Get the top-left pixel position of a tile at (row, col)."""
    x = layout.margin_x + (col * (layout.tile_size + layout.gap_size))
    y = layout.margin_y + (row * (layout.tile_size + layout.gap_size))
    return (x, y)


def get_tile_at_mouse(mouse_pos: tuple[int, int]) -> Position | None:
    """Get tile coordinates from mouse position, or None if not on a tile."""
    x, y = mouse_pos
    grid_x = x - layout.margin_x
    grid_y = y - layout.margin_y
    cell_size = layout.tile_size + layout.gap_size
    
    col = grid_x // cell_size
    row = grid_y // cell_size
    
    relative_x = grid_x % cell_size
    relative_y = grid_y % cell_size
    
    if relative_x > layout.tile_size or relative_y > layout.tile_size:
        return None
    
    if 0 <= row < GRID_COUNT and 0 <= col < GRID_COUNT:
        return (row, col)
    
    return None


def get_wall_at_mouse(mouse_pos: tuple[int, int], wall_orientation: str) -> Wall | None:
    """
    Get wall position from mouse position.
    Returns a Wall tuple or None if not valid.
    """
    x, y = mouse_pos
    grid_x = x - layout.margin_x
    grid_y = y - layout.margin_y
    cell_size = layout.tile_size + layout.gap_size
    
    # Calculate which cell intersection we're near
    col = grid_x // cell_size
    row = grid_y // cell_size
    
    # Get position within the cell
    relative_x = grid_x % cell_size
    relative_y = grid_y % cell_size
    
    # Determine if we're in a gap area
    in_horizontal_gap = relative_y > layout.tile_size
    in_vertical_gap = relative_x > layout.tile_size
    
    # For horizontal walls, check if we're in the horizontal gap between rows
    if wall_orientation == 'H':
        if in_horizontal_gap and 0 <= row < GRID_COUNT - 1 and 0 <= col < GRID_COUNT - 1:
            return ((row, col), 'H')
        # Also allow placing when hovering over the tile, using the gap below
        elif not in_horizontal_gap and 0 <= row < GRID_COUNT - 1 and 0 <= col < GRID_COUNT - 1:
            return ((row, col), 'H')
    
    # For vertical walls, check if we're in the vertical gap between columns
    elif wall_orientation == 'V':
        if in_vertical_gap and 0 <= row < GRID_COUNT - 1 and 0 <= col < GRID_COUNT - 1:
            return ((row, col), 'V')
        elif not in_vertical_gap and 0 <= row < GRID_COUNT - 1 and 0 <= col < GRID_COUNT - 1:
            return ((row, col), 'V')
    
    return None


def get_wall_rect(wall: Wall) -> pygame.Rect:
    """Get the pygame Rect for drawing a wall."""
    (row, col), orientation = wall
    
    if orientation == 'H':
        # Horizontal wall: spans gap between row and row+1, covers 2 tiles + gap
        x = layout.margin_x + col * (layout.tile_size + layout.gap_size)
        y = layout.margin_y + row * (layout.tile_size + layout.gap_size) + layout.tile_size + (layout.gap_size - layout.wall_thickness) // 2
        width = 2 * layout.tile_size + layout.gap_size
        height = layout.wall_thickness
    else:  # Vertical
        # Vertical wall: spans gap between col and col+1, covers 2 tiles + gap
        x = layout.margin_x + col * (layout.tile_size + layout.gap_size) + layout.tile_size + (layout.gap_size - layout.wall_thickness) // 2
        y = layout.margin_y + row * (layout.tile_size + layout.gap_size)
        width = layout.wall_thickness
        height = 2 * layout.tile_size + layout.gap_size
    
    return pygame.Rect(x, y, width, height)


def draw_button(text: str, rect: pygame.Rect, hover: bool = False, 
                style: str = "normal", disabled: bool = False) -> None:
    """Draw a modern styled button with text.
    
    Styles: 'normal', 'primary', 'success', 'danger'
    """
    # Define style colors
    styles = {
        "normal": {
            "bg": BUTTON_COLOR,
            "hover": BUTTON_HOVER_COLOR,
            "border": BUTTON_BORDER,
            "text": TEXT_COLOR
        },
        "primary": {
            "bg": (45, 85, 130),
            "hover": (55, 105, 160),
            "border": (80, 140, 200),
            "text": TEXT_COLOR
        },
        "success": {
            "bg": (40, 90, 50),
            "hover": (50, 120, 60),
            "border": (80, 180, 100),
            "text": TEXT_COLOR
        },
        "danger": {
            "bg": (120, 40, 40),
            "hover": (150, 50, 50),
            "border": (200, 80, 80),
            "text": TEXT_COLOR
        }
    }
    
    s = styles.get(style, styles["normal"])
    
    if disabled:
        bg_color = (60, 55, 50)
        border_color = (80, 75, 70)
        text_color = (120, 115, 110)
    else:
        bg_color = s["hover"] if hover else s["bg"]
        border_color = s["border"] if hover else (s["border"][0]//2, s["border"][1]//2, s["border"][2]//2)
        text_color = s["text"]
    
    # Draw shadow
    if not disabled:
        shadow_rect = pygame.Rect(rect.x + 3, rect.y + 3, rect.width, rect.height)
        shadow = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(shadow, (0, 0, 0, 40), shadow.get_rect(), border_radius=10)
        screen.blit(shadow, shadow_rect.topleft)
    
    # Draw button background with subtle gradient
    btn_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.rect(btn_surf, bg_color, btn_surf.get_rect(), border_radius=10)
    
    # Add highlight at top
    if not disabled:
        highlight = pygame.Surface((rect.width - 4, 2), pygame.SRCALPHA)
        highlight.fill((*[min(255, c + 30) for c in bg_color[:3]], 80))
        btn_surf.blit(highlight, (2, 2))
    
    screen.blit(btn_surf, rect.topleft)
    
    # Draw border
    pygame.draw.rect(screen, border_color, rect, 2, border_radius=10)
    
    # Draw text with slight shadow
    if not disabled:
        shadow_surf = font.render(text, True, (0, 0, 0))
        shadow_rect = shadow_surf.get_rect(center=(rect.centerx + 1, rect.centery + 1))
        shadow_surf.set_alpha(60)
        screen.blit(shadow_surf, shadow_rect)
    
    text_surf = font.render(text, True, text_color)
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)


# --- GAME RENDERING ---

def draw_board(game_state: GameState, valid_moves: list[Position], 
               wall_preview: Wall | None = None, wall_valid: bool = True) -> None:
    """Draw the complete game board with enhanced visuals."""
    global animation_tick
    animation_tick = (animation_tick + 1) % 60
    
    screen.fill(BG_COLOR)
    
    # Draw board frame/border
    board_rect = pygame.Rect(layout.margin_x - 10, layout.margin_y - 10, 
                             layout.board_size + 20, layout.board_size + 20)
    pygame.draw.rect(screen, (40, 35, 25), board_rect, border_radius=8)
    pygame.draw.rect(screen, (80, 70, 50), board_rect, 3, border_radius=8)
    
    # Draw tiles
    for row in range(GRID_COUNT):
        for col in range(GRID_COUNT):
            x, y = get_tile_topleft(row, col)
            screen.blit(tile_img, (x, y))
    
    # Draw valid move highlights with pulsing effect
    pulse = abs(30 - animation_tick) / 30.0  # 0 to 1 pulse
    for pos in valid_moves:
        x, y = get_tile_topleft(*pos)
        
        # Draw outer glow
        glow_size = int(8 + 4 * pulse)
        glow = pygame.Surface((layout.tile_size + glow_size*2, layout.tile_size + glow_size*2), pygame.SRCALPHA)
        pygame.draw.rect(glow, (*HIGHLIGHT_COLOR, int(40 + 30 * pulse)), 
                        glow.get_rect(), border_radius=12)
        screen.blit(glow, (x - glow_size, y - glow_size))
        
        # Draw highlight
        highlight = pygame.Surface((layout.tile_size, layout.tile_size), pygame.SRCALPHA)
        alpha = int(80 + 40 * pulse)
        pygame.draw.rect(highlight, (*HIGHLIGHT_COLOR, alpha), highlight.get_rect(), border_radius=8)
        pygame.draw.rect(highlight, (*HIGHLIGHT_PULSE, int(150 + 50 * pulse)), 
                        highlight.get_rect(), 3, border_radius=8)
        screen.blit(highlight, (x, y))
    
    # Draw placed walls with 3D effect
    for wall in game_state.walls:
        rect = get_wall_rect(wall)
        # Shadow
        shadow_rect = pygame.Rect(rect.x + 2, rect.y + 2, rect.width, rect.height)
        pygame.draw.rect(screen, (20, 15, 10), shadow_rect, border_radius=3)
        # Main wall
        pygame.draw.rect(screen, WALL_COLOR, rect, border_radius=3)
        # Highlight edge
        pygame.draw.rect(screen, WALL_HIGHLIGHT, rect, 1, border_radius=3)
    
    # Draw wall preview
    if wall_preview:
        rect = get_wall_rect(wall_preview)
        color = WALL_PREVIEW_COLOR if wall_valid else WALL_INVALID_COLOR
        preview_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(preview_surf, (*color, 180), preview_surf.get_rect(), border_radius=3)
        pygame.draw.rect(preview_surf, (*color, 255), preview_surf.get_rect(), 2, border_radius=3)
        screen.blit(preview_surf, rect.topleft)
    
    # Draw pawns with enhanced 3D effect
    p1_center = get_tile_center(*game_state.player1_pos)
    p2_center = get_tile_center(*game_state.player2_pos)
    
    # Determine if pawns should glow (current player)
    p1_glow = game_state.current_player == 1
    p2_glow = game_state.current_player == 2
    
    for center, color, glow_color, is_glowing in [
        (p1_center, PLAYER1_COLOR, PLAYER1_GLOW, p1_glow),
        (p2_center, PLAYER2_COLOR, PLAYER2_GLOW, p2_glow)
    ]:
        # Glow effect for current player
        if is_glowing:
            glow_radius = layout.pawn_radius + 8 + int(4 * pulse)
            glow_surf = pygame.Surface((glow_radius*2 + 20, glow_radius*2 + 20), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*glow_color, int(60 + 30 * pulse)), 
                             (glow_radius + 10, glow_radius + 10), glow_radius)
            screen.blit(glow_surf, (center[0] - glow_radius - 10, center[1] - glow_radius - 10))
        
        # Shadow
        pygame.draw.circle(screen, (20, 15, 10), (center[0] + 3, center[1] + 3), layout.pawn_radius)
        
        # Main pawn body
        pygame.draw.circle(screen, color, center, layout.pawn_radius)
        
        # Inner highlight (3D effect)
        highlight_pos = (center[0] - layout.pawn_radius//3, center[1] - layout.pawn_radius//3)
        highlight_surf = pygame.Surface((layout.pawn_radius*2, layout.pawn_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(highlight_surf, (*[min(255, c + 60) for c in color], 100),
                          (layout.pawn_radius//2, layout.pawn_radius//2), layout.pawn_radius//2)
        screen.blit(highlight_surf, (center[0] - layout.pawn_radius, center[1] - layout.pawn_radius))
        
        # Outer ring
        pygame.draw.circle(screen, (255, 255, 255), center, layout.pawn_radius, 3)
        pygame.draw.circle(screen, (*[max(0, c - 40) for c in color], ), center, layout.pawn_radius - 3, 2)


def draw_ui(game_state: GameState, mode: str, wall_mode: bool, wall_orientation: str,
            status_message: str = "") -> tuple[pygame.Rect, pygame.Rect]:
    """Draw game UI elements and return button rects (save_rect, menu_rect)."""
    
    sw, sh = layout.screen_width, layout.screen_height
    
    # Top panel for turn indicator
    panel_width = min(400, int(sw * 0.35))
    panel_height = max(35, int(45 * sh / DEFAULT_SCREEN_HEIGHT))
    top_panel = pygame.Rect(layout.margin_x - 10, 5, panel_width, panel_height)
    draw_panel(top_panel, 200)
    
    turn_color = PLAYER1_COLOR if game_state.current_player == 1 else PLAYER2_COLOR
    turn_text = f"{'Player 1' if game_state.current_player == 1 else 'Player 2'}'s turn"
    if mode == MODE_PVE and game_state.current_player == 2:
        turn_text = "AI thinking..."
        turn_color = PLAYER2_COLOR
    
    # Turn indicator dot
    dot_y = top_panel.centery
    pygame.draw.circle(screen, turn_color, (layout.margin_x + 10, dot_y), 8)
    pygame.draw.circle(screen, (255, 255, 255), (layout.margin_x + 10, dot_y), 8, 2)
    
    turn_surf = font.render(turn_text, True, TEXT_COLOR)
    screen.blit(turn_surf, (layout.margin_x + 30, top_panel.y + (panel_height - turn_surf.get_height()) // 2))
    
    # Mode panel (top right)
    mode_panel_width = min(370, int(sw * 0.3))
    mode_panel = pygame.Rect(sw - mode_panel_width - 10, 5, mode_panel_width, panel_height)
    draw_panel(mode_panel, 200)
    
    if wall_mode:
        mode_text = f"Wall Mode: {wall_orientation}"
        mode_color = ACCENT_COLOR
    else:
        mode_text = "Move Mode"
        mode_color = HIGHLIGHT_COLOR
    
    mode_surf = font.render(mode_text, True, mode_color)
    screen.blit(mode_surf, (mode_panel.x + 10, mode_panel.y + (panel_height - mode_surf.get_height()) // 2))
    
    # Bottom left panel - Player info
    info_panel_width = min(350, int(sw * 0.3))
    info_panel_height = max(70, int(85 * sh / DEFAULT_SCREEN_HEIGHT))
    info_panel = pygame.Rect(layout.margin_x - 10, sh - info_panel_height - 10, info_panel_width, info_panel_height)
    draw_panel(info_panel, 200)
    
    # Player 1 info with icon
    p1_y = info_panel.y + info_panel_height // 3
    pygame.draw.circle(screen, PLAYER1_COLOR, (layout.margin_x + 15, p1_y), 10)
    pygame.draw.circle(screen, (255, 255, 255), (layout.margin_x + 15, p1_y), 10, 2)
    p1_text = f"Player 1: {game_state.player1_walls} walls"
    p1_surf = font.render(p1_text, True, PLAYER1_GLOW if game_state.current_player == 1 else TEXT_SECONDARY)
    screen.blit(p1_surf, (layout.margin_x + 35, p1_y - p1_surf.get_height() // 2))
    
    # Player 2 info with icon
    p2_y = info_panel.y + info_panel_height * 2 // 3
    pygame.draw.circle(screen, PLAYER2_COLOR, (layout.margin_x + 15, p2_y), 10)
    pygame.draw.circle(screen, (255, 255, 255), (layout.margin_x + 15, p2_y), 10, 2)
    p2_text = f"Player 2: {game_state.player2_walls} walls"
    p2_surf = font.render(p2_text, True, PLAYER2_GLOW if game_state.current_player == 2 else TEXT_SECONDARY)
    screen.blit(p2_surf, (layout.margin_x + 35, p2_y - p2_surf.get_height() // 2))
    
    # Bottom center - Instructions panel
    inst_panel_width = min(500, int(sw * 0.4))
    inst_panel_height = max(40, int(50 * sh / DEFAULT_SCREEN_HEIGHT))
    inst_panel = pygame.Rect(sw // 2 - inst_panel_width // 2, sh - inst_panel_height - 10, 
                             inst_panel_width, inst_panel_height)
    draw_panel(inst_panel, 180)
    
    if wall_mode:
        inst_text = "Click to place wall  •  R: rotate  •  W: cancel"
    else:
        inst_text = "Click tile to move  •  W: wall mode"
    inst_surf = small_font.render(inst_text, True, TEXT_SECONDARY)
    inst_rect = inst_surf.get_rect(center=inst_panel.center)
    screen.blit(inst_surf, inst_rect)
    
    # Bottom right - Save and Menu buttons
    mouse_pos = pygame.mouse.get_pos()
    btn_width = max(80, int(105 * sw / DEFAULT_SCREEN_WIDTH))
    btn_height = max(35, int(42 * sh / DEFAULT_SCREEN_HEIGHT))
    btn_spacing = 10
    
    save_rect = pygame.Rect(sw - btn_width * 2 - btn_spacing - 10, sh - btn_height - 10, btn_width, btn_height)
    menu_rect = pygame.Rect(sw - btn_width - 10, sh - btn_height - 10, btn_width, btn_height)
    
    draw_button("Save", save_rect, save_rect.collidepoint(mouse_pos), style="success")
    draw_button("Menu", menu_rect, menu_rect.collidepoint(mouse_pos), style="danger")
    
    # Status message with animation
    if status_message:
        status_width = min(220, int(sw * 0.18))
        status_height = max(35, int(40 * sh / DEFAULT_SCREEN_HEIGHT))
        status_panel = pygame.Rect(sw - status_width - 10, sh - btn_height - status_height - 20, 
                                   status_width, status_height)
        pygame.draw.rect(screen, (30, 80, 40), status_panel, border_radius=8)
        pygame.draw.rect(screen, (80, 180, 100), status_panel, 2, border_radius=8)
        status_surf = font.render(status_message, True, (150, 255, 150))
        status_rect = status_surf.get_rect(center=status_panel.center)
        screen.blit(status_surf, status_rect)
    
    return save_rect, menu_rect


def draw_menu() -> tuple[pygame.Rect, pygame.Rect, pygame.Rect]:
    """Draw the main menu and return button rects (pvp, pve, load)."""
    screen.fill(BG_COLOR)
    
    sw, sh = layout.screen_width, layout.screen_height
    
    # Decorative background pattern
    for i in range(0, sw, 100):
        for j in range(0, sh, 100):
            alpha = 10 + ((i + j) % 20)
            dot_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(dot_surf, (80, 70, 50, alpha), (2, 2), 2)
            screen.blit(dot_surf, (i, j))
    
    # Main content panel - scale with window
    panel_width = min(600, int(sw * 0.7))
    panel_height = min(700, int(sh * 0.65))
    panel_y = int(sh * 0.25)
    panel_rect = pygame.Rect(sw // 2 - panel_width // 2, panel_y, panel_width, panel_height)
    panel = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
    pygame.draw.rect(panel, (35, 30, 25, 240), panel.get_rect(), border_radius=20)
    screen.blit(panel, panel_rect.topleft)
    pygame.draw.rect(screen, PANEL_BORDER, panel_rect, 3, border_radius=20)
    
    # Title with glow effect
    title_y = panel_y + int(panel_height * 0.1)
    title_glow = title_font.render("QUORIDOR", True, ACCENT_COLOR)
    title_glow.set_alpha(30)
    for offset in [(2, 2), (-2, -2), (2, -2), (-2, 2)]:
        screen.blit(title_glow, title_glow.get_rect(center=(sw // 2 + offset[0], title_y + offset[1])))
    
    title = title_font.render("QUORIDOR", True, ACCENT_COLOR)
    title_rect = title.get_rect(center=(sw // 2, title_y))
    screen.blit(title, title_rect)
    
    # Decorative line under title
    line_y = title_y + int(title.get_height() * 0.7)
    pygame.draw.line(screen, PANEL_BORDER, (sw // 2 - 150, line_y), 
                    (sw // 2 + 150, line_y), 2)
    pygame.draw.circle(screen, ACCENT_COLOR, (sw // 2, line_y), 5)
    
    # Subtitle
    subtitle_y = line_y + 30
    subtitle = font.render("A Strategic Board Game", True, TEXT_SECONDARY)
    sub_rect = subtitle.get_rect(center=(sw // 2, subtitle_y))
    screen.blit(subtitle, sub_rect)
    
    # Buttons with icons - scale sizes
    btn_width = min(360, int(panel_width * 0.8))
    btn_height = max(50, int(65 * sh / DEFAULT_SCREEN_HEIGHT))
    btn_spacing = max(15, int(20 * sh / DEFAULT_SCREEN_HEIGHT))
    
    btn_start_y = subtitle_y + 60
    
    pvp_rect = pygame.Rect(sw // 2 - btn_width // 2, btn_start_y, btn_width, btn_height)
    pve_rect = pygame.Rect(sw // 2 - btn_width // 2, btn_start_y + btn_height + btn_spacing, btn_width, btn_height)
    load_rect = pygame.Rect(sw // 2 - btn_width // 2, btn_start_y + (btn_height + btn_spacing) * 2, btn_width, btn_height)
    
    mouse_pos = pygame.mouse.get_pos()
    draw_button("Player vs Player", pvp_rect, pvp_rect.collidepoint(mouse_pos), style="primary")
    draw_button("Player vs AI", pve_rect, pve_rect.collidepoint(mouse_pos), style="primary")
    draw_button("Load Game", load_rect, load_rect.collidepoint(mouse_pos))
    
    # Instructions panel - only show if there's enough space
    inst_start_y = load_rect.bottom + 30
    available_space = panel_rect.bottom - inst_start_y - 20
    
    if available_space > 120:
        inst_panel_width = min(500, int(panel_width * 0.9))
        inst_panel_height = min(180, available_space)
        inst_panel = pygame.Rect(sw // 2 - inst_panel_width // 2, inst_start_y, 
                                 inst_panel_width, inst_panel_height)
        inst_bg = pygame.Surface((inst_panel.width, inst_panel.height), pygame.SRCALPHA)
        pygame.draw.rect(inst_bg, (30, 25, 20, 180), inst_bg.get_rect(), border_radius=12)
        screen.blit(inst_bg, inst_panel.topleft)
        pygame.draw.rect(screen, (60, 50, 40), inst_panel, 1, border_radius=12)
        
        # Instructions header
        header = font.render("How to Play", True, ACCENT_COLOR)
        screen.blit(header, (inst_panel.x + 20, inst_panel.y + 10))
        
        instructions = [
            "• Move your pawn to the opposite side to win",
            "• Place walls (W key) to block your opponent",
            "• Each player has 10 walls to use",
            "• You cannot completely block a path"
        ]
        
        y_offset = inst_panel.y + 50
        line_height = max(25, int(30 * sh / DEFAULT_SCREEN_HEIGHT))
        for line in instructions:
            if y_offset + line_height > inst_panel.bottom - 10:
                break
            inst_surf = small_font.render(line, True, TEXT_SECONDARY)
            screen.blit(inst_surf, (inst_panel.x + 25, y_offset))
            y_offset += line_height
    
    return pvp_rect, pve_rect, load_rect


def draw_game_over(winner: int) -> pygame.Rect:
    """Draw game over screen and return restart button rect."""
    sw, sh = layout.screen_width, layout.screen_height
    
    # Semi-transparent overlay with gradient
    overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
    for y in range(sh):
        alpha = int(180 + 40 * (y / sh))
        pygame.draw.line(overlay, (0, 0, 0, min(255, alpha)), (0, y), (sw, y))
    screen.blit(overlay, (0, 0))
    
    # Victory panel - scale with screen
    panel_width = min(400, int(sw * 0.5))
    panel_height = min(240, int(sh * 0.3))
    panel_rect = pygame.Rect(sw // 2 - panel_width // 2, sh // 2 - panel_height // 2, 
                             panel_width, panel_height)
    panel = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
    pygame.draw.rect(panel, (30, 25, 20, 250), panel.get_rect(), border_radius=20)
    screen.blit(panel, panel_rect.topleft)
    
    winner_color = PLAYER1_COLOR if winner == 1 else PLAYER2_COLOR
    glow_color = PLAYER1_GLOW if winner == 1 else PLAYER2_GLOW
    pygame.draw.rect(screen, winner_color, panel_rect, 3, border_radius=20)
    
    # Winner text    
    winner_text = f"Player {winner} Wins!"
    winner_surf = large_font.render(winner_text, True, glow_color)
    winner_rect = winner_surf.get_rect(center=(sw // 2, panel_rect.centery - panel_height // 6))
    screen.blit(winner_surf, winner_rect)
    
    # Restart button
    btn_width = min(200, int(panel_width * 0.6))
    btn_height = max(45, int(55 * sh / DEFAULT_SCREEN_HEIGHT))
    restart_rect = pygame.Rect(sw // 2 - btn_width // 2, panel_rect.centery + panel_height // 6, 
                               btn_width, btn_height)
    mouse_pos = pygame.mouse.get_pos()
    draw_button("Main Menu", restart_rect, restart_rect.collidepoint(mouse_pos))
    
    return restart_rect


def draw_difficulty_select() -> tuple[pygame.Rect, pygame.Rect, pygame.Rect, pygame.Rect]:
    """Draw difficulty selection screen and return button rects."""
    screen.fill(BG_COLOR)
    
    sw, sh = layout.screen_width, layout.screen_height
    
    # Decorative background
    for i in range(0, sw, 100):
        for j in range(0, sh, 100):
            alpha = 10 + ((i + j) % 20)
            dot_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(dot_surf, (80, 70, 50, alpha), (2, 2), 2)
            screen.blit(dot_surf, (i, j))
    
    # Main panel - scale with window
    panel_width = min(560, int(sw * 0.7))
    panel_height = min(700, int(sh * 0.75))
    panel_y = int(sh * 0.12)
    panel_rect = pygame.Rect(sw // 2 - panel_width // 2, panel_y, panel_width, panel_height)
    panel = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
    pygame.draw.rect(panel, (35, 30, 25, 240), panel.get_rect(), border_radius=20)
    screen.blit(panel, panel_rect.topleft)
    pygame.draw.rect(screen, PANEL_BORDER, panel_rect, 3, border_radius=20)
    
    # Title
    title_y = panel_y + int(panel_height * 0.08)
    title = large_font.render("SELECT DIFFICULTY", True, ACCENT_COLOR)
    title_rect = title.get_rect(center=(sw // 2, title_y))
    screen.blit(title, title_rect)
    
    # Decorative line
    line_y = title_y + int(title.get_height() * 0.8)
    pygame.draw.line(screen, PANEL_BORDER, (sw // 2 - 150, line_y), 
                    (sw // 2 + 150, line_y), 2)
    
    mouse_pos = pygame.mouse.get_pos()
    
    # Difficulty buttons with enhanced styling - scale sizes
    btn_width = min(360, int(panel_width * 0.8))
    btn_height = max(80, int(110 * sh / DEFAULT_SCREEN_HEIGHT))
    btn_spacing = max(20, int(30 * sh / DEFAULT_SCREEN_HEIGHT))
    back_btn_height = max(45, int(55 * sh / DEFAULT_SCREEN_HEIGHT))
    
    btn_start_y = line_y + 50
    
    easy_rect = pygame.Rect(sw // 2 - btn_width // 2, btn_start_y, btn_width, btn_height)
    medium_rect = pygame.Rect(sw // 2 - btn_width // 2, btn_start_y + btn_height + btn_spacing, btn_width, btn_height)
    hard_rect = pygame.Rect(sw // 2 - btn_width // 2, btn_start_y + (btn_height + btn_spacing) * 2, btn_width, btn_height)
    
    back_btn_width = min(240, int(btn_width * 0.7))
    back_y = min(hard_rect.bottom + btn_spacing + 20, panel_rect.bottom - back_btn_height - 20)
    back_rect = pygame.Rect(sw // 2 - back_btn_width // 2, back_y, back_btn_width, back_btn_height)
    
    # Helper function for difficulty buttons
    def draw_difficulty_button(rect, label, subtitle, colors, hover):
        bg_color = colors["hover"] if hover else colors["bg"]
        border_color = colors["border"]
        text_color = colors["text"]
        
        # Shadow
        shadow = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(shadow, (0, 0, 0, 50), shadow.get_rect(), border_radius=15)
        screen.blit(shadow, (rect.x + 4, rect.y + 4))
        
        # Button background
        pygame.draw.rect(screen, bg_color, rect, border_radius=15)
        pygame.draw.rect(screen, border_color, rect, 3, border_radius=15)
        
        # Glow on hover
        if hover:
            glow = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            pygame.draw.rect(glow, (*border_color, 30), glow.get_rect(), border_radius=15)
            screen.blit(glow, rect.topleft)
        
        # Label
        label_surf = large_font.render(label, True, text_color)
        label_rect = label_surf.get_rect(center=(rect.centerx, rect.centery - rect.height // 8))
        screen.blit(label_surf, label_rect)
        
        # Subtitle
        sub_surf = small_font.render(subtitle, True, (*text_color[:3],))
        sub_surf.set_alpha(180)
        sub_rect = sub_surf.get_rect(center=(rect.centerx, rect.centery + rect.height // 4))
        screen.blit(sub_surf, sub_rect)
    
    # Easy button - Green theme
    easy_colors = {
        "bg": (35, 75, 40),
        "hover": (45, 95, 50),
        "border": (80, 180, 100),
        "text": (150, 255, 160)
    }
    draw_difficulty_button(easy_rect, "Easy", "Relaxed gameplay", 
                          easy_colors, easy_rect.collidepoint(mouse_pos))
    
    # Medium button - Golden/Orange theme
    medium_colors = {
        "bg": (85, 70, 30),
        "hover": (105, 90, 40),
        "border": (200, 170, 80),
        "text": (255, 220, 120)
    }
    draw_difficulty_button(medium_rect, "Medium", "Balanced challenge",
                          medium_colors, medium_rect.collidepoint(mouse_pos))
    
    # Hard button - Red theme
    hard_colors = {
        "bg": (80, 35, 35),
        "hover": (100, 45, 45),
        "border": (200, 80, 80),
        "text": (255, 130, 130)
    }
    draw_difficulty_button(hard_rect, "Hard", "Expert AI opponent",
                          hard_colors, hard_rect.collidepoint(mouse_pos))
    
    # Back button
    draw_button("Back", back_rect, back_rect.collidepoint(mouse_pos))
    
    return easy_rect, medium_rect, hard_rect, back_rect


# Game mode for difficulty selection
MODE_DIFFICULTY = 'difficulty'


def draw_load_screen(scroll_offset: int = 0) -> tuple[list[tuple[pygame.Rect, any]], pygame.Rect]:
    """Draw load game screen and return (save_button_rects, back_rect)."""
    screen.fill(BG_COLOR)
    
    sw, sh = layout.screen_width, layout.screen_height
    
    # Decorative background
    for i in range(0, sw, 100):
        for j in range(0, sh, 100):
            alpha = 10 + ((i + j) % 20)
            dot_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(dot_surf, (80, 70, 50, alpha), (2, 2), 2)
            screen.blit(dot_surf, (i, j))
    
    # Main panel - scale with window
    panel_width = min(700, int(sw * 0.85))
    panel_height = sh - 120
    panel_rect = pygame.Rect(sw // 2 - panel_width // 2, 60, panel_width, panel_height)
    panel = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
    pygame.draw.rect(panel, (35, 30, 25, 240), panel.get_rect(), border_radius=20)
    screen.blit(panel, panel_rect.topleft)
    pygame.draw.rect(screen, PANEL_BORDER, panel_rect, 3, border_radius=20)
    
    # Title
    title_y = 100
    title = large_font.render("LOAD GAME", True, ACCENT_COLOR)
    title_rect = title.get_rect(center=(sw // 2, title_y))
    screen.blit(title, title_rect)
    
    # Decorative line
    line_y = title_y + int(title.get_height() * 0.7)
    pygame.draw.line(screen, PANEL_BORDER, (sw // 2 - 150, line_y), 
                    (sw // 2 + 150, line_y), 2)
    
    mouse_pos = pygame.mouse.get_pos()
    
    # Get save files
    saves = get_save_files()
    
    save_buttons = []
    
    if not saves:
        # No saves message with icon
        empty_panel_width = min(400, int(panel_width * 0.7))
        empty_panel_height = min(150, int(sh * 0.15))
        empty_panel = pygame.Rect(sw // 2 - empty_panel_width // 2, sh // 2 - empty_panel_height // 2, 
                                  empty_panel_width, empty_panel_height)
        pygame.draw.rect(screen, (40, 35, 30), empty_panel, border_radius=15)
        pygame.draw.rect(screen, (60, 55, 45), empty_panel, 2, border_radius=15)
        
        no_saves = font.render("No saved games found", True, TEXT_SECONDARY)
        screen.blit(no_saves, no_saves.get_rect(center=(sw // 2, empty_panel.centery - 15)))
        
        hint = small_font.render("Start a game and click Save to create one", True, (120, 115, 100))
        screen.blit(hint, hint.get_rect(center=(sw // 2, empty_panel.centery + 15)))
    else:
        # Draw save file buttons - scale sizes
        y_start = line_y + 40
        button_height = max(60, int(80 * sh / DEFAULT_SCREEN_HEIGHT))
        button_spacing = max(8, int(12 * sh / DEFAULT_SCREEN_HEIGHT))
        btn_width = min(600, int(panel_width * 0.9))
        
        # Calculate max visible based on available space
        back_btn_height = max(45, int(55 * sh / DEFAULT_SCREEN_HEIGHT))
        available_height = sh - y_start - back_btn_height - 100
        max_visible = max(3, int(available_height / (button_height + button_spacing)))
        
        visible_saves = saves[scroll_offset:scroll_offset + max_visible]
        
        for i, (filepath, display_name, timestamp) in enumerate(visible_saves):
            y = y_start + i * (button_height + button_spacing)
            rect = pygame.Rect(sw // 2 - btn_width // 2, y, btn_width, button_height)
            
            hover = rect.collidepoint(mouse_pos)
            
            # Shadow
            if hover:
                shadow = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                pygame.draw.rect(shadow, (0, 0, 0, 40), shadow.get_rect(), border_radius=12)
                screen.blit(shadow, (rect.x + 3, rect.y + 3))
            
            # Button background
            bg_color = (55, 50, 45) if hover else (45, 40, 35)
            pygame.draw.rect(screen, bg_color, rect, border_radius=12)
            
            # Border
            border_color = ACCENT_COLOR if hover else (70, 65, 55)
            pygame.draw.rect(screen, border_color, rect, 2, border_radius=12)
            
            # Display name (mode + difficulty)
            name_surf = font.render(display_name, True, TEXT_COLOR if hover else TEXT_SECONDARY)
            screen.blit(name_surf, (rect.x + 55, rect.y + button_height // 4 - name_surf.get_height() // 2))
            
            # Timestamp
            time_surf = small_font.render(f"{timestamp}", True, (140, 135, 125))
            screen.blit(time_surf, (rect.x + 55, rect.y + button_height * 3 // 4 - time_surf.get_height() // 2))
            
            # Filename (smaller, right side)
            filename = filepath.name
            file_surf = tiny_font.render(filename, True, (100, 95, 85))
            file_rect = file_surf.get_rect(right=rect.right - 15, centery=rect.centery)
            screen.blit(file_surf, file_rect)
            
            save_buttons.append((rect, filepath))
        
        # Scroll indicators
        if scroll_offset > 0:
            up_indicator = pygame.Rect(sw // 2 - 100, y_start - 25, 200, 25)
            pygame.draw.rect(screen, (50, 45, 40), up_indicator, border_radius=5)
            up_text = small_font.render("Scroll up for more", True, ACCENT_COLOR)
            screen.blit(up_text, up_text.get_rect(center=up_indicator.center))
        
        if scroll_offset + max_visible < len(saves):
            down_y = y_start + max_visible * (button_height + button_spacing) + 5
            down_indicator = pygame.Rect(sw // 2 - 100, down_y, 200, 25)
            pygame.draw.rect(screen, (50, 45, 40), down_indicator, border_radius=5)
            down_text = small_font.render("Scroll down for more", True, ACCENT_COLOR)
            screen.blit(down_text, down_text.get_rect(center=down_indicator.center))
    
    # Back button
    back_btn_width = min(240, int(panel_width * 0.4))
    back_btn_height = max(45, int(55 * sh / DEFAULT_SCREEN_HEIGHT))
    back_rect = pygame.Rect(sw // 2 - back_btn_width // 2, sh - back_btn_height - 25, 
                            back_btn_width, back_btn_height)
    draw_button("Back", back_rect, back_rect.collidepoint(mouse_pos))
    
    return save_buttons, back_rect


# --- MAIN GAME LOOP ---

def main():
    """Main game loop."""
    global screen
    
    game_mode = MODE_MENU
    game_state = GameState()
    ai_player = None
    selected_difficulty = 'medium'
    
    # UI state
    wall_mode = False
    wall_orientation = 'H'
    valid_moves = []
    
    # Save/Load state
    status_message = ""
    status_message_timer = 0
    load_scroll_offset = 0
    
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        # Update status message timer
        if status_message_timer > 0:
            status_message_timer -= 1
            if status_message_timer == 0:
                status_message = ""
        
        # --- EVENT HANDLING ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Handle window resize
            if event.type == pygame.VIDEORESIZE:
                screen = handle_resize(event.w, event.h)
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if game_mode == MODE_DIFFICULTY:
                        game_mode = MODE_MENU
                    elif game_mode == MODE_LOAD:
                        game_mode = MODE_MENU
                        load_scroll_offset = 0
                    elif game_mode in [MODE_PVP, MODE_PVE]:
                        game_mode = MODE_MENU
                        game_state = GameState()
                    else:
                        running = False
                
                if game_mode in [MODE_PVP, MODE_PVE] and not game_state.is_terminal():
                    # Don't allow input during AI turn
                    if game_mode == MODE_PVE and game_state.current_player == 2:
                        continue
                    
                    if event.key == pygame.K_w:
                        wall_mode = not wall_mode
                        valid_moves = [] if wall_mode else game_state.get_valid_moves(game_state.current_player)
                    
                    if event.key == pygame.K_r and wall_mode:
                        wall_orientation = 'V' if wall_orientation == 'H' else 'H'
            
            # Scroll wheel for load screen
            if event.type == pygame.MOUSEWHEEL and game_mode == MODE_LOAD:
                saves = get_save_files()
                max_visible = 8
                max_offset = max(0, len(saves) - max_visible)
                if event.y > 0:  # Scroll up
                    load_scroll_offset = max(0, load_scroll_offset - 1)
                elif event.y < 0:  # Scroll down
                    load_scroll_offset = min(max_offset, load_scroll_offset + 1)
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Menu
                if game_mode == MODE_MENU:
                    pvp_rect, pve_rect, load_rect = draw_menu()
                    if pvp_rect.collidepoint(mouse_pos):
                        game_mode = MODE_PVP
                        game_state = GameState()
                        valid_moves = game_state.get_valid_moves(game_state.current_player)
                    elif pve_rect.collidepoint(mouse_pos):
                        game_mode = MODE_DIFFICULTY  # Go to difficulty selection
                    elif load_rect.collidepoint(mouse_pos):
                        game_mode = MODE_LOAD
                        load_scroll_offset = 0
                
                # Difficulty selection
                elif game_mode == MODE_DIFFICULTY:
                    easy_rect, medium_rect, hard_rect, back_rect = draw_difficulty_select()
                    if easy_rect.collidepoint(mouse_pos):
                        selected_difficulty = 'easy'
                        game_mode = MODE_PVE
                        game_state = GameState()
                        ai_player = load_ai_player(player=2, difficulty=selected_difficulty)
                        valid_moves = game_state.get_valid_moves(game_state.current_player)
                    elif medium_rect.collidepoint(mouse_pos):
                        selected_difficulty = 'medium'
                        game_mode = MODE_PVE
                        game_state = GameState()
                        ai_player = load_ai_player(player=2, difficulty=selected_difficulty)
                        valid_moves = game_state.get_valid_moves(game_state.current_player)
                    elif hard_rect.collidepoint(mouse_pos):
                        selected_difficulty = 'hard'
                        game_mode = MODE_PVE
                        game_state = GameState()
                        ai_player = load_ai_player(player=2, difficulty=selected_difficulty)
                        valid_moves = game_state.get_valid_moves(game_state.current_player)
                    elif back_rect.collidepoint(mouse_pos):
                        game_mode = MODE_MENU
                
                # Load game screen
                elif game_mode == MODE_LOAD:
                    save_buttons, back_rect = draw_load_screen(load_scroll_offset)
                    if back_rect.collidepoint(mouse_pos):
                        game_mode = MODE_MENU
                        load_scroll_offset = 0
                    else:
                        for rect, filepath in save_buttons:
                            if rect.collidepoint(mouse_pos):
                                try:
                                    game_state, loaded_mode, loaded_difficulty = load_game(filepath)
                                    game_mode = loaded_mode
                                    selected_difficulty = loaded_difficulty
                                    valid_moves = game_state.get_valid_moves(game_state.current_player)
                                    wall_mode = False
                                    
                                    # Load AI if PVE mode
                                    if game_mode == MODE_PVE and loaded_difficulty:
                                        ai_player = load_ai_player(player=2, difficulty=loaded_difficulty)
                                    
                                    status_message = "Game Loaded!"
                                    status_message_timer = 120  # ~2 seconds at 60 FPS
                                    load_scroll_offset = 0
                                except Exception as e:
                                    print(f"Error loading save: {e}")
                                    status_message = "Load Failed!"
                                    status_message_timer = 120
                                break
                
                # Game over - restart
                elif game_state.is_terminal():
                    restart_rect = draw_game_over(game_state.winner)
                    if restart_rect.collidepoint(mouse_pos):
                        game_mode = MODE_MENU
                        game_state = GameState()
                
                # In-game actions
                elif game_mode in [MODE_PVP, MODE_PVE]:
                    # Don't allow input during AI turn
                    if game_mode == MODE_PVE and game_state.current_player == 2:
                        continue
                    
                    # Check Save and Menu buttons first
                    save_rect, menu_rect = draw_ui(game_state, game_mode, wall_mode, wall_orientation, status_message)
                    
                    if save_rect.collidepoint(mouse_pos) and not game_state.is_terminal():
                        # Save the game
                        difficulty = selected_difficulty if game_mode == MODE_PVE else None
                        save_game(game_state, game_mode, difficulty)
                        status_message = "Game Saved!"
                        status_message_timer = 120  # ~2 seconds at 60 FPS
                    elif menu_rect.collidepoint(mouse_pos):
                        # Return to main menu
                        game_mode = MODE_MENU
                        game_state = GameState()
                        wall_mode = False
                        status_message = ""
                    elif wall_mode:
                        # Wall placement
                        wall = get_wall_at_mouse(mouse_pos, wall_orientation)
                        if wall and game_state.is_valid_wall_placement(wall):
                            game_state.apply_action(('wall', wall))
                            wall_mode = False
                            valid_moves = game_state.get_valid_moves(game_state.current_player)
                    else:
                        # Pawn movement
                        clicked_tile = get_tile_at_mouse(mouse_pos)
                        if clicked_tile and clicked_tile in valid_moves:
                            game_state.apply_action(('move', clicked_tile))
                            valid_moves = game_state.get_valid_moves(game_state.current_player)
        
        # --- AI TURN ---
        if (game_mode == MODE_PVE and game_state.current_player == 2 
            and not game_state.is_terminal()):
            # Draw current state first
            draw_board(game_state, [], None, True)
            draw_ui(game_state, game_mode, wall_mode, wall_orientation, status_message)
            pygame.display.flip()
            
            # Get AI move
            action = ai_player.get_action(game_state)
            if action:
                game_state.apply_action(action)
                valid_moves = game_state.get_valid_moves(game_state.current_player)
        
        # --- RENDERING ---
        if game_mode == MODE_MENU:
            draw_menu()
        
        elif game_mode == MODE_DIFFICULTY:
            draw_difficulty_select()
        
        elif game_mode == MODE_LOAD:
            draw_load_screen(load_scroll_offset)
        
        elif game_mode in [MODE_PVP, MODE_PVE]:
            # Get wall preview if in wall mode
            wall_preview = None
            wall_valid = True
            if wall_mode and not game_state.is_terminal():
                if not (game_mode == MODE_PVE and game_state.current_player == 2):
                    wall_preview = get_wall_at_mouse(mouse_pos, wall_orientation)
                    if wall_preview:
                        wall_valid = game_state.is_valid_wall_placement(wall_preview)
            
            draw_board(game_state, valid_moves if not wall_mode else [], wall_preview, wall_valid)
            draw_ui(game_state, game_mode, wall_mode, wall_orientation, status_message)
            
            if game_state.is_terminal():
                draw_game_over(game_state.winner)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()