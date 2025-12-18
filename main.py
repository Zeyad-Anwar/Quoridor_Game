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
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Quoridor")
clock = pygame.time.Clock()

# Fonts - Modern font setup
try:
    title_font = pygame.font.Font(None, 96)  # Larger title
    large_font = pygame.font.Font(None, 72)
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 28)
    tiny_font = pygame.font.Font(None, 22)
except:
    title_font = pygame.font.SysFont('arial', 72)
    large_font = pygame.font.SysFont('arial', 54)
    font = pygame.font.SysFont('arial', 28)
    small_font = pygame.font.SysFont('arial', 22)
    tiny_font = pygame.font.SysFont('arial', 18)

# Load Assets
raw_tile_img = pygame.image.load('assets/tile.png')
tile_img = pygame.transform.scale(raw_tile_img, (TILE_SIZE, TILE_SIZE))

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
    x = MARGIN_X + (col * (TILE_SIZE + GAP_SIZE)) + TILE_SIZE // 2
    y = MARGIN_Y + (row * (TILE_SIZE + GAP_SIZE)) + TILE_SIZE // 2
    return (x, y)


def get_tile_topleft(row: int, col: int) -> tuple[int, int]:
    """Get the top-left pixel position of a tile at (row, col)."""
    x = MARGIN_X + (col * (TILE_SIZE + GAP_SIZE))
    y = MARGIN_Y + (row * (TILE_SIZE + GAP_SIZE))
    return (x, y)


def get_tile_at_mouse(mouse_pos: tuple[int, int]) -> Position | None:
    """Get tile coordinates from mouse position, or None if not on a tile."""
    x, y = mouse_pos
    grid_x = x - MARGIN_X
    grid_y = y - MARGIN_Y
    cell_size = TILE_SIZE + GAP_SIZE
    
    col = grid_x // cell_size
    row = grid_y // cell_size
    
    relative_x = grid_x % cell_size
    relative_y = grid_y % cell_size
    
    if relative_x > TILE_SIZE or relative_y > TILE_SIZE:
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
    grid_x = x - MARGIN_X
    grid_y = y - MARGIN_Y
    cell_size = TILE_SIZE + GAP_SIZE
    
    # Calculate which cell intersection we're near
    col = grid_x // cell_size
    row = grid_y // cell_size
    
    # Get position within the cell
    relative_x = grid_x % cell_size
    relative_y = grid_y % cell_size
    
    # Determine if we're in a gap area
    in_horizontal_gap = relative_y > TILE_SIZE
    in_vertical_gap = relative_x > TILE_SIZE
    
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
        x = MARGIN_X + col * (TILE_SIZE + GAP_SIZE)
        y = MARGIN_Y + row * (TILE_SIZE + GAP_SIZE) + TILE_SIZE + (GAP_SIZE - WALL_THICKNESS) // 2
        width = 2 * TILE_SIZE + GAP_SIZE
        height = WALL_THICKNESS
    else:  # Vertical
        # Vertical wall: spans gap between col and col+1, covers 2 tiles + gap
        x = MARGIN_X + col * (TILE_SIZE + GAP_SIZE) + TILE_SIZE + (GAP_SIZE - WALL_THICKNESS) // 2
        y = MARGIN_Y + row * (TILE_SIZE + GAP_SIZE)
        width = WALL_THICKNESS
        height = 2 * TILE_SIZE + GAP_SIZE
    
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
    board_width = GRID_COUNT * TILE_SIZE + (GRID_COUNT - 1) * GAP_SIZE
    board_rect = pygame.Rect(MARGIN_X - 10, MARGIN_Y - 10, board_width + 20, board_width + 20)
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
        glow = pygame.Surface((TILE_SIZE + glow_size*2, TILE_SIZE + glow_size*2), pygame.SRCALPHA)
        pygame.draw.rect(glow, (*HIGHLIGHT_COLOR, int(40 + 30 * pulse)), 
                        glow.get_rect(), border_radius=12)
        screen.blit(glow, (x - glow_size, y - glow_size))
        
        # Draw highlight
        highlight = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
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
            glow_radius = PAWN_RADIUS + 8 + int(4 * pulse)
            glow_surf = pygame.Surface((glow_radius*2 + 20, glow_radius*2 + 20), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*glow_color, int(60 + 30 * pulse)), 
                             (glow_radius + 10, glow_radius + 10), glow_radius)
            screen.blit(glow_surf, (center[0] - glow_radius - 10, center[1] - glow_radius - 10))
        
        # Shadow
        pygame.draw.circle(screen, (20, 15, 10), (center[0] + 3, center[1] + 3), PAWN_RADIUS)
        
        # Main pawn body
        pygame.draw.circle(screen, color, center, PAWN_RADIUS)
        
        # Inner highlight (3D effect)
        highlight_pos = (center[0] - PAWN_RADIUS//3, center[1] - PAWN_RADIUS//3)
        highlight_surf = pygame.Surface((PAWN_RADIUS*2, PAWN_RADIUS*2), pygame.SRCALPHA)
        pygame.draw.circle(highlight_surf, (*[min(255, c + 60) for c in color], 100),
                          (PAWN_RADIUS//2, PAWN_RADIUS//2), PAWN_RADIUS//2)
        screen.blit(highlight_surf, (center[0] - PAWN_RADIUS, center[1] - PAWN_RADIUS))
        
        # Outer ring
        pygame.draw.circle(screen, (255, 255, 255), center, PAWN_RADIUS, 3)
        pygame.draw.circle(screen, (*[max(0, c - 40) for c in color], ), center, PAWN_RADIUS - 3, 2)


def draw_ui(game_state: GameState, mode: str, wall_mode: bool, wall_orientation: str,
            status_message: str = "") -> tuple[pygame.Rect, pygame.Rect]:
    """Draw game UI elements and return button rects (save_rect, menu_rect)."""
    
    # Top panel for turn indicator
    top_panel = pygame.Rect(MARGIN_X - 10, 5, 400, 45)
    draw_panel(top_panel, 200)
    
    turn_color = PLAYER1_COLOR if game_state.current_player == 1 else PLAYER2_COLOR
    turn_text = f"{'Player 1' if game_state.current_player == 1 else 'Player 2'}'s turn"
    if mode == MODE_PVE and game_state.current_player == 2:
        turn_text = "AI thinking..."
        turn_color = PLAYER2_COLOR
    
    # Turn indicator dot
    pygame.draw.circle(screen, turn_color, (MARGIN_X + 10, 27), 8)
    pygame.draw.circle(screen, (255, 255, 255), (MARGIN_X + 10, 27), 8, 2)
    
    turn_surf = font.render(turn_text, True, TEXT_COLOR)
    screen.blit(turn_surf, (MARGIN_X + 30, 15))
    
    # Mode panel (top right)
    mode_panel = pygame.Rect(SCREEN_WIDTH - 380, 5, 370, 45)
    draw_panel(mode_panel, 200)
    
    if wall_mode:
        mode_text = f"Wall Mode: {wall_orientation}"
        mode_color = ACCENT_COLOR
    else:
        mode_text = "Move Mode"
        mode_color = HIGHLIGHT_COLOR
    
    mode_surf = font.render(mode_text, True, mode_color)
    screen.blit(mode_surf, (SCREEN_WIDTH - 370, 15))
    
    # Bottom left panel - Player info
    info_panel = pygame.Rect(MARGIN_X - 10, SCREEN_HEIGHT - 95, 350, 85)
    draw_panel(info_panel, 200)
    
    # Player 1 info with icon
    pygame.draw.circle(screen, PLAYER1_COLOR, (MARGIN_X + 15, SCREEN_HEIGHT - 70), 10)
    pygame.draw.circle(screen, (255, 255, 255), (MARGIN_X + 15, SCREEN_HEIGHT - 70), 10, 2)
    p1_text = f"Player 1: {game_state.player1_walls} walls"
    p1_surf = font.render(p1_text, True, PLAYER1_GLOW if game_state.current_player == 1 else TEXT_SECONDARY)
    screen.blit(p1_surf, (MARGIN_X + 35, SCREEN_HEIGHT - 80))
    
    # Player 2 info with icon
    pygame.draw.circle(screen, PLAYER2_COLOR, (MARGIN_X + 15, SCREEN_HEIGHT - 35), 10)
    pygame.draw.circle(screen, (255, 255, 255), (MARGIN_X + 15, SCREEN_HEIGHT - 35), 10, 2)
    p2_text = f"Player 2: {game_state.player2_walls} walls"
    p2_surf = font.render(p2_text, True, PLAYER2_GLOW if game_state.current_player == 2 else TEXT_SECONDARY)
    screen.blit(p2_surf, (MARGIN_X + 35, SCREEN_HEIGHT - 45))
    
    # Bottom center - Instructions panel
    inst_panel = pygame.Rect(SCREEN_WIDTH // 2 - 250, SCREEN_HEIGHT - 60, 500, 50)
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
    save_rect = pygame.Rect(SCREEN_WIDTH - 230, SCREEN_HEIGHT - 55, 105, 42)
    menu_rect = pygame.Rect(SCREEN_WIDTH - 115, SCREEN_HEIGHT - 55, 105, 42)
    
    draw_button("Save", save_rect, save_rect.collidepoint(mouse_pos), style="success")
    draw_button("Menu", menu_rect, menu_rect.collidepoint(mouse_pos), style="danger")
    
    # Status message with animation
    if status_message:
        status_panel = pygame.Rect(SCREEN_WIDTH - 230, SCREEN_HEIGHT - 105, 220, 40)
        pygame.draw.rect(screen, (30, 80, 40), status_panel, border_radius=8)
        pygame.draw.rect(screen, (80, 180, 100), status_panel, 2, border_radius=8)
        status_surf = font.render(status_message, True, (150, 255, 150))
        status_rect = status_surf.get_rect(center=status_panel.center)
        screen.blit(status_surf, status_rect)
    
    return save_rect, menu_rect


def draw_menu() -> tuple[pygame.Rect, pygame.Rect, pygame.Rect]:
    """Draw the main menu and return button rects (pvp, pve, load)."""
    screen.fill(BG_COLOR)
    
    # Decorative background pattern
    for i in range(0, SCREEN_WIDTH, 100):
        for j in range(0, SCREEN_HEIGHT, 100):
            alpha = 10 + ((i + j) % 20)
            dot_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(dot_surf, (80, 70, 50, alpha), (2, 2), 2)
            screen.blit(dot_surf, (i, j))
    
    # Main content panel
    panel_rect = pygame.Rect(SCREEN_WIDTH // 2 - 300, 350, 600, 700)
    panel = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
    pygame.draw.rect(panel, (35, 30, 25, 240), panel.get_rect(), border_radius=20)
    screen.blit(panel, panel_rect.topleft)
    pygame.draw.rect(screen, PANEL_BORDER, panel_rect, 3, border_radius=20)
    
    # Title with glow effect
    title_glow = title_font.render("QUORIDOR", True, ACCENT_COLOR)
    title_glow.set_alpha(30)
    for offset in [(2, 2), (-2, -2), (2, -2), (-2, 2)]:
        screen.blit(title_glow, title_glow.get_rect(center=(SCREEN_WIDTH // 2 + offset[0], 420 + offset[1])))
    
    title = title_font.render("QUORIDOR", True, ACCENT_COLOR)
    title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 420))
    screen.blit(title, title_rect)
    
    # Decorative line under title
    line_y = 470
    pygame.draw.line(screen, PANEL_BORDER, (SCREEN_WIDTH // 2 - 150, line_y), 
                    (SCREEN_WIDTH // 2 + 150, line_y), 2)
    pygame.draw.circle(screen, ACCENT_COLOR, (SCREEN_WIDTH // 2, line_y), 5)
    
    # Subtitle
    subtitle = font.render("A Strategic Board Game", True, TEXT_SECONDARY)
    sub_rect = subtitle.get_rect(center=(SCREEN_WIDTH // 2, 500))
    screen.blit(subtitle, sub_rect)
    
    # Buttons with icons
    pvp_rect = pygame.Rect(SCREEN_WIDTH // 2 - 180, 560, 360, 65)
    pve_rect = pygame.Rect(SCREEN_WIDTH // 2 - 180, 645, 360, 65)
    load_rect = pygame.Rect(SCREEN_WIDTH // 2 - 180, 730, 360, 65)
    
    mouse_pos = pygame.mouse.get_pos()
    draw_button("Player vs Player", pvp_rect, pvp_rect.collidepoint(mouse_pos), style="primary")
    draw_button("Player vs AI", pve_rect, pve_rect.collidepoint(mouse_pos), style="primary")
    draw_button("Load Game", load_rect, load_rect.collidepoint(mouse_pos))
    
    # Instructions panel
    inst_panel = pygame.Rect(SCREEN_WIDTH // 2 - 250, 830, 500, 180)
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
    for line in instructions:
        inst_surf = small_font.render(line, True, TEXT_SECONDARY)
        screen.blit(inst_surf, (inst_panel.x + 25, y_offset))
        y_offset += 30
    
    return pvp_rect, pve_rect, load_rect


def draw_game_over(winner: int) -> pygame.Rect:
    """Draw game over screen and return restart button rect."""
    # Semi-transparent overlay with gradient
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    for y in range(SCREEN_HEIGHT):
        alpha = int(180 + 40 * (y / SCREEN_HEIGHT))
        pygame.draw.line(overlay, (0, 0, 0, min(255, alpha)), (0, y), (SCREEN_WIDTH, y))
    screen.blit(overlay, (0, 0))
    
    # Victory panel
    panel_rect = pygame.Rect(SCREEN_WIDTH // 2 - 200, SCREEN_HEIGHT // 2 - 120, 400, 240)
    panel = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
    pygame.draw.rect(panel, (30, 25, 20, 250), panel.get_rect(), border_radius=20)
    screen.blit(panel, panel_rect.topleft)
    
    winner_color = PLAYER1_COLOR if winner == 1 else PLAYER2_COLOR
    glow_color = PLAYER1_GLOW if winner == 1 else PLAYER2_GLOW
    pygame.draw.rect(screen, winner_color, panel_rect, 3, border_radius=20)
    
    # Winner text    
    winner_text = f"Player {winner} Wins!"
    winner_surf = large_font.render(winner_text, True, glow_color)
    winner_rect = winner_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 10))
    screen.blit(winner_surf, winner_rect)
    
    # Restart button
    restart_rect = pygame.Rect(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 50, 200, 55)
    mouse_pos = pygame.mouse.get_pos()
    draw_button("Main Menu", restart_rect, restart_rect.collidepoint(mouse_pos))
    
    return restart_rect


def draw_difficulty_select() -> tuple[pygame.Rect, pygame.Rect, pygame.Rect, pygame.Rect]:
    """Draw difficulty selection screen and return button rects."""
    screen.fill(BG_COLOR)
    
    # Decorative background
    for i in range(0, SCREEN_WIDTH, 100):
        for j in range(0, SCREEN_HEIGHT, 100):
            alpha = 10 + ((i + j) % 20)
            dot_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(dot_surf, (80, 70, 50, alpha), (2, 2), 2)
            screen.blit(dot_surf, (i, j))
    
    # Main panel
    panel_rect = pygame.Rect(SCREEN_WIDTH // 2 - 280, 200, 560, 700)
    panel = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
    pygame.draw.rect(panel, (35, 30, 25, 240), panel.get_rect(), border_radius=20)
    screen.blit(panel, panel_rect.topleft)
    pygame.draw.rect(screen, PANEL_BORDER, panel_rect, 3, border_radius=20)
    
    # Title
    title = large_font.render("SELECT DIFFICULTY", True, ACCENT_COLOR)
    title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 280))
    screen.blit(title, title_rect)
    
    # Decorative line
    pygame.draw.line(screen, PANEL_BORDER, (SCREEN_WIDTH // 2 - 150, 320), 
                    (SCREEN_WIDTH // 2 + 150, 320), 2)
    
    mouse_pos = pygame.mouse.get_pos()
    
    # Difficulty buttons with enhanced styling
    easy_rect = pygame.Rect(SCREEN_WIDTH // 2 - 180, 370, 360, 110)
    medium_rect = pygame.Rect(SCREEN_WIDTH // 2 - 180, 510, 360, 110)
    hard_rect = pygame.Rect(SCREEN_WIDTH // 2 - 180, 650, 360, 110)
    back_rect = pygame.Rect(SCREEN_WIDTH // 2 - 120, 800, 240, 55)
    
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
        label_rect = label_surf.get_rect(center=(rect.centerx, rect.centery - 12))
        screen.blit(label_surf, label_rect)
        
        # Subtitle
        sub_surf = small_font.render(subtitle, True, (*text_color[:3],))
        sub_surf.set_alpha(180)
        sub_rect = sub_surf.get_rect(center=(rect.centerx, rect.centery + 25))
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
    
    # Decorative background
    for i in range(0, SCREEN_WIDTH, 100):
        for j in range(0, SCREEN_HEIGHT, 100):
            alpha = 10 + ((i + j) % 20)
            dot_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(dot_surf, (80, 70, 50, alpha), (2, 2), 2)
            screen.blit(dot_surf, (i, j))
    
    # Main panel
    panel_rect = pygame.Rect(SCREEN_WIDTH // 2 - 350, 60, 700, SCREEN_HEIGHT - 120)
    panel = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
    pygame.draw.rect(panel, (35, 30, 25, 240), panel.get_rect(), border_radius=20)
    screen.blit(panel, panel_rect.topleft)
    pygame.draw.rect(screen, PANEL_BORDER, panel_rect, 3, border_radius=20)
    
    # Title
    title = large_font.render("LOAD GAME", True, ACCENT_COLOR)
    title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 120))
    screen.blit(title, title_rect)
    
    # Decorative line
    pygame.draw.line(screen, PANEL_BORDER, (SCREEN_WIDTH // 2 - 150, 160), 
                    (SCREEN_WIDTH // 2 + 150, 160), 2)
    
    mouse_pos = pygame.mouse.get_pos()
    
    # Get save files
    saves = get_save_files()
    
    save_buttons = []
    
    if not saves:
        # No saves message with icon
        empty_panel = pygame.Rect(SCREEN_WIDTH // 2 - 200, 350, 400, 150)
        pygame.draw.rect(screen, (40, 35, 30), empty_panel, border_radius=15)
        pygame.draw.rect(screen, (60, 55, 45), empty_panel, 2, border_radius=15)
        
        no_saves = font.render("No saved games found", True, TEXT_SECONDARY)
        screen.blit(no_saves, no_saves.get_rect(center=(SCREEN_WIDTH // 2, 450)))
        
        hint = small_font.render("Start a game and click Save to create one", True, (120, 115, 100))
        screen.blit(hint, hint.get_rect(center=(SCREEN_WIDTH // 2, 480)))
    else:
        # Draw save file buttons
        y_start = 200
        button_height = 80
        button_spacing = 12
        max_visible = 12
        
        visible_saves = saves[scroll_offset:scroll_offset + max_visible]
        
        for i, (filepath, display_name, timestamp) in enumerate(visible_saves):
            y = y_start + i * (button_height + button_spacing)
            rect = pygame.Rect(SCREEN_WIDTH // 2 - 300, y, 600, button_height)
            
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
            screen.blit(name_surf, (rect.x + 55, rect.y + 15))
            
            # Timestamp
            time_surf = small_font.render(f"{timestamp}", True, (140, 135, 125))
            screen.blit(time_surf, (rect.x + 55, rect.y + 48))
            
            # Filename (smaller, right side)
            filename = filepath.name
            file_surf = tiny_font.render(filename, True, (100, 95, 85))
            file_rect = file_surf.get_rect(right=rect.right - 15, centery=rect.centery)
            screen.blit(file_surf, file_rect)
            
            save_buttons.append((rect, filepath))
        
        # Scroll indicators
        if scroll_offset > 0:
            up_indicator = pygame.Rect(SCREEN_WIDTH // 2 - 100, 175, 200, 25)
            pygame.draw.rect(screen, (50, 45, 40), up_indicator, border_radius=5)
            up_text = small_font.render("Scroll up for more", True, ACCENT_COLOR)
            screen.blit(up_text, up_text.get_rect(center=up_indicator.center))
        
        if scroll_offset + max_visible < len(saves):
            down_y = y_start + max_visible * (button_height + button_spacing) + 5
            down_indicator = pygame.Rect(SCREEN_WIDTH // 2 - 100, down_y, 200, 25)
            pygame.draw.rect(screen, (50, 45, 40), down_indicator, border_radius=5)
            down_text = small_font.render("Scroll down for more", True, ACCENT_COLOR)
            screen.blit(down_text, down_text.get_rect(center=down_indicator.center))
    
    # Back button
    back_rect = pygame.Rect(SCREEN_WIDTH // 2 - 120, SCREEN_HEIGHT - 80, 240, 55)
    draw_button("Back", back_rect, back_rect.collidepoint(mouse_pos))
    
    return save_buttons, back_rect


# --- MAIN GAME LOOP ---

def main():
    """Main game loop."""
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