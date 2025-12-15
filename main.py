"""
Quoridor Game - Main Entry Point
A strategic board game with pygame implementation.
"""
from constants import *
from game import GameState, Wall, Position
from AI.mcts import AIPlayer
import pygame
import sys


# --- SETUP ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Quoridor")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)
small_font = pygame.font.Font(None, 28)

# Load Assets
raw_tile_img = pygame.image.load('assets/tile.png')
tile_img = pygame.transform.scale(raw_tile_img, (TILE_SIZE, TILE_SIZE))


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


def draw_button(text: str, rect: pygame.Rect, hover: bool = False) -> None:
    """Draw a button with text."""
    color = BUTTON_HOVER_COLOR if hover else BUTTON_COLOR
    pygame.draw.rect(screen, color, rect, border_radius=8)
    pygame.draw.rect(screen, TEXT_COLOR, rect, 2, border_radius=8)
    
    text_surf = font.render(text, True, TEXT_COLOR)
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)


# --- GAME RENDERING ---

def draw_board(game_state: GameState, valid_moves: list[Position], 
               wall_preview: Wall | None = None, wall_valid: bool = True) -> None:
    """Draw the complete game board."""
    screen.fill(BG_COLOR)
    
    # Draw tiles
    for row in range(GRID_COUNT):
        for col in range(GRID_COUNT):
            x, y = get_tile_topleft(row, col)
            screen.blit(tile_img, (x, y))
    
    # Draw valid move highlights
    for pos in valid_moves:
        x, y = get_tile_topleft(*pos)
        highlight = pygame.Surface((TILE_SIZE, TILE_SIZE))
        highlight.set_alpha(100)
        highlight.fill(HIGHLIGHT_COLOR)
        screen.blit(highlight, (x, y))
    
    # Draw placed walls
    for wall in game_state.walls:
        rect = get_wall_rect(wall)
        pygame.draw.rect(screen, WALL_COLOR, rect, border_radius=3)
    
    # Draw wall preview
    if wall_preview:
        rect = get_wall_rect(wall_preview)
        color = WALL_PREVIEW_COLOR if wall_valid else WALL_INVALID_COLOR
        preview_surf = pygame.Surface((rect.width, rect.height))
        preview_surf.set_alpha(180)
        preview_surf.fill(color)
        screen.blit(preview_surf, rect.topleft)
    
    # Draw pawns
    p1_center = get_tile_center(*game_state.player1_pos)
    p2_center = get_tile_center(*game_state.player2_pos)
    
    pygame.draw.circle(screen, PLAYER1_COLOR, p1_center, PAWN_RADIUS)
    pygame.draw.circle(screen, (255, 255, 255), p1_center, PAWN_RADIUS, 3)
    
    pygame.draw.circle(screen, PLAYER2_COLOR, p2_center, PAWN_RADIUS)
    pygame.draw.circle(screen, (255, 255, 255), p2_center, PAWN_RADIUS, 3)


def draw_ui(game_state: GameState, mode: str, wall_mode: bool, wall_orientation: str) -> None:
    """Draw game UI elements."""
    # Player info
    p1_text = f"Player 1 (Red): {game_state.player1_walls} walls"
    p2_text = f"Player 2 (Blue): {game_state.player2_walls} walls"
    
    p1_surf = small_font.render(p1_text, True, PLAYER1_COLOR)
    p2_surf = small_font.render(p2_text, True, PLAYER2_COLOR)
    
    screen.blit(p1_surf, (MARGIN_X, SCREEN_HEIGHT - 80))
    screen.blit(p2_surf, (MARGIN_X, SCREEN_HEIGHT - 50))
    
    # Current turn indicator
    turn_text = f"{'Player 1' if game_state.current_player == 1 else 'Player 2'}'s turn"
    if mode == MODE_PVE and game_state.current_player == 2:
        turn_text = "AI thinking..."
    turn_surf = font.render(turn_text, True, TEXT_COLOR)
    screen.blit(turn_surf, (MARGIN_X, 10))
    
    # Mode indicator
    mode_text = "Press W to toggle wall mode" if not wall_mode else f"Wall mode: {wall_orientation} (R to rotate)"
    mode_surf = small_font.render(mode_text, True, TEXT_COLOR)
    screen.blit(mode_surf, (SCREEN_WIDTH - 300, 10))
    
    # Instructions
    if wall_mode:
        inst_text = "Click to place wall | R: rotate | W: cancel"
    else:
        inst_text = "Click highlighted tile to move | W: wall mode"
    inst_surf = small_font.render(inst_text, True, TEXT_COLOR)
    screen.blit(inst_surf, (SCREEN_WIDTH - 400, SCREEN_HEIGHT - 25))


def draw_menu() -> tuple[pygame.Rect, pygame.Rect]:
    """Draw the main menu and return button rects."""
    screen.fill(BG_COLOR)
    
    # Title
    title = font.render("QUORIDOR", True, TEXT_COLOR)
    title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 150))
    screen.blit(title, title_rect)
    
    # Subtitle
    subtitle = small_font.render("A Strategic Board Game", True, TEXT_COLOR)
    sub_rect = subtitle.get_rect(center=(SCREEN_WIDTH // 2, 200))
    screen.blit(subtitle, sub_rect)
    
    # Buttons
    pvp_rect = pygame.Rect(SCREEN_WIDTH // 2 - 120, 300, 240, 60)
    pve_rect = pygame.Rect(SCREEN_WIDTH // 2 - 120, 400, 240, 60)
    
    mouse_pos = pygame.mouse.get_pos()
    draw_button("Player vs Player", pvp_rect, pvp_rect.collidepoint(mouse_pos))
    draw_button("Player vs AI", pve_rect, pve_rect.collidepoint(mouse_pos))
    
    # Instructions
    instructions = [
        "How to Play:",
        "- Move your pawn to the opposite side to win",
        "- Place walls (W key) to block your opponent",
        "- Each player has 10 walls",
        "- You cannot completely block a player's path"
    ]
    
    y_offset = 500
    for line in instructions:
        inst_surf = small_font.render(line, True, TEXT_COLOR)
        inst_rect = inst_surf.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
        screen.blit(inst_surf, inst_rect)
        y_offset += 30
    
    return pvp_rect, pve_rect


def draw_game_over(winner: int) -> pygame.Rect:
    """Draw game over screen and return restart button rect."""
    # Semi-transparent overlay
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    overlay.set_alpha(200)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))
    
    # Winner text
    winner_text = f"Player {winner} Wins!"
    winner_surf = font.render(winner_text, True, PLAYER1_COLOR if winner == 1 else PLAYER2_COLOR)
    winner_rect = winner_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
    screen.blit(winner_surf, winner_rect)
    
    # Restart button
    restart_rect = pygame.Rect(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 30, 200, 50)
    mouse_pos = pygame.mouse.get_pos()
    draw_button("Main Menu", restart_rect, restart_rect.collidepoint(mouse_pos))
    
    return restart_rect


# --- MAIN GAME LOOP ---

def main():
    """Main game loop."""
    game_mode = MODE_MENU
    game_state = GameState()
    ai_player = None
    
    # UI state
    wall_mode = False
    wall_orientation = 'H'
    valid_moves = []
    
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        # --- EVENT HANDLING ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if game_mode in [MODE_PVP, MODE_PVE]:
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
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Menu
                if game_mode == MODE_MENU:
                    pvp_rect, pve_rect = draw_menu()
                    if pvp_rect.collidepoint(mouse_pos):
                        game_mode = MODE_PVP
                        game_state = GameState()
                        valid_moves = game_state.get_valid_moves(game_state.current_player)
                    elif pve_rect.collidepoint(mouse_pos):
                        game_mode = MODE_PVE
                        game_state = GameState()
                        ai_player = AIPlayer(player=2)
                        valid_moves = game_state.get_valid_moves(game_state.current_player)
                
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
                    
                    if wall_mode:
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
            draw_ui(game_state, game_mode, wall_mode, wall_orientation)
            pygame.display.flip()
            
            # Get AI move
            action = ai_player.get_action(game_state)
            if action:
                game_state.apply_action(action)
                valid_moves = game_state.get_valid_moves(game_state.current_player)
        
        # --- RENDERING ---
        if game_mode == MODE_MENU:
            draw_menu()
        
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
            draw_ui(game_state, game_mode, wall_mode, wall_orientation)
            
            if game_state.is_terminal():
                draw_game_over(game_state.winner)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()