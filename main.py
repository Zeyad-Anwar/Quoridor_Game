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

