import os
import torch
import pygame
import sys

from constants import *

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


def draw_difficulty_select() -> tuple[pygame.Rect, pygame.Rect, pygame.Rect, pygame.Rect]:
    """Draw difficulty selection screen and return button rects."""
    screen.fill(BG_COLOR)
    
    # Title
    title = font.render("SELECT DIFFICULTY", True, TEXT_COLOR)
    title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 150))
    screen.blit(title, title_rect)
    
    mouse_pos = pygame.mouse.get_pos()
    
    # Difficulty buttons with colors
    easy_rect = pygame.Rect(SCREEN_WIDTH // 2 - 120, 250, 240, 60)
    medium_rect = pygame.Rect(SCREEN_WIDTH // 2 - 120, 340, 240, 60)
    hard_rect = pygame.Rect(SCREEN_WIDTH // 2 - 120, 430, 240, 60)
    back_rect = pygame.Rect(SCREEN_WIDTH // 2 - 120, 550, 240, 50)
    
    # Draw buttons with difficulty-specific colors
    # Easy - Green tint
    easy_color = (50, 120, 50) if easy_rect.collidepoint(mouse_pos) else (40, 90, 40)
    pygame.draw.rect(screen, easy_color, easy_rect, border_radius=8)
    pygame.draw.rect(screen, (100, 200, 100), easy_rect, 2, border_radius=8)
    easy_text = font.render("Easy", True, (150, 255, 150))
    screen.blit(easy_text, easy_text.get_rect(center=easy_rect.center))
    
    # Easy description
    easy_desc = small_font.render("Casual play with randomness", True, (150, 200, 150))
    screen.blit(easy_desc, easy_desc.get_rect(center=(SCREEN_WIDTH // 2, 295)))
    
    # Medium - Yellow/Orange tint
    medium_color = (120, 100, 30) if medium_rect.collidepoint(mouse_pos) else (90, 75, 25)
    pygame.draw.rect(screen, medium_color, medium_rect, border_radius=8)
    pygame.draw.rect(screen, (200, 180, 80), medium_rect, 2, border_radius=8)
    medium_text = font.render("Medium", True, (255, 220, 100))
    screen.blit(medium_text, medium_text.get_rect(center=medium_rect.center))
    
    # Medium description
    medium_desc = small_font.render("Balanced challenge", True, (200, 180, 100))
    screen.blit(medium_desc, medium_desc.get_rect(center=(SCREEN_WIDTH // 2, 385)))
    
    # Hard - Red tint
    hard_color = (120, 40, 40) if hard_rect.collidepoint(mouse_pos) else (90, 30, 30)
    pygame.draw.rect(screen, hard_color, hard_rect, border_radius=8)
    pygame.draw.rect(screen, (200, 80, 80), hard_rect, 2, border_radius=8)
    hard_text = font.render("Hard", True, (255, 120, 120))
    screen.blit(hard_text, hard_text.get_rect(center=hard_rect.center))
    
    # Hard description
    hard_desc = small_font.render("Near-optimal play", True, (200, 120, 120))
    screen.blit(hard_desc, hard_desc.get_rect(center=(SCREEN_WIDTH // 2, 475)))
    
    # Back button
    draw_button("Back", back_rect, back_rect.collidepoint(mouse_pos))
    
    return easy_rect, medium_rect, hard_rect, back_rect
