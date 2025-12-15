"""
State Encoding for AlphaZero Neural Network
Converts GameState to tensor representation.

Input Planes (7 channels, each 9x9):
- Plane 0: Player 1 position (one-hot)
- Plane 1: Player 2 position (one-hot)
- Plane 2: Horizontal walls (8x8 padded to 9x9)
- Plane 3: Vertical walls (8x8 padded to 9x9)
- Plane 4: Current player (all 1s if player 1, all 0s if player 2)
- Plane 5: Player 1 walls remaining (normalized, broadcast)
- Plane 6: Player 2 walls remaining (normalized, broadcast)
"""
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game import GameState

from constants import GRID_COUNT, INPUT_CHANNELS


def encode_state(game_state: 'GameState') -> np.ndarray:
    """
    Encode a game state as a tensor for the neural network.
    
    Args:
        game_state: The current game state
    
    Returns:
        Numpy array of shape (INPUT_CHANNELS, GRID_COUNT, GRID_COUNT)
    """
    planes = np.zeros((INPUT_CHANNELS, GRID_COUNT, GRID_COUNT), dtype=np.float32)
    
    # Plane 0: Player 1 position (one-hot)
    p1_row, p1_col = game_state.player1_pos
    planes[0, p1_row, p1_col] = 1.0
    
    # Plane 1: Player 2 position (one-hot)
    p2_row, p2_col = game_state.player2_pos
    planes[1, p2_row, p2_col] = 1.0
    
    # Plane 2 & 3: Walls
    for wall in game_state.walls:
        (row, col), orientation = wall
        if orientation == 'H':
            # Horizontal wall at (row, col) - mark on plane 2
            planes[2, row, col] = 1.0
            # Also mark the extended position to show wall spans 2 tiles
            if col + 1 < GRID_COUNT:
                planes[2, row, col + 1] = 1.0
        else:  # 'V'
            # Vertical wall at (row, col) - mark on plane 3
            planes[3, row, col] = 1.0
            # Also mark the extended position
            if row + 1 < GRID_COUNT:
                planes[3, row + 1, col] = 1.0
    
    # Plane 4: Current player indicator
    # 1.0 if current player is player 1, 0.0 if player 2
    if game_state.current_player == 1:
        planes[4, :, :] = 1.0
    
    # Plane 5: Player 1 walls remaining (normalized to [0, 1])
    p1_walls_norm = game_state.player1_walls / 10.0
    planes[5, :, :] = p1_walls_norm
    
    # Plane 6: Player 2 walls remaining (normalized to [0, 1])
    p2_walls_norm = game_state.player2_walls / 10.0
    planes[6, :, :] = p2_walls_norm
    
    return planes


def encode_state_batch(game_states: list['GameState']) -> np.ndarray:
    """
    Encode multiple game states as a batch.
    
    Args:
        game_states: List of game states
    
    Returns:
        Numpy array of shape (batch_size, INPUT_CHANNELS, GRID_COUNT, GRID_COUNT)
    """
    batch = np.stack([encode_state(gs) for gs in game_states], axis=0)
    return batch


def encode_state_augmented(game_state: 'GameState') -> list[np.ndarray]:
    """
    Encode state with data augmentation (rotations/reflections).
    For Quoridor, we can use horizontal reflection (left-right symmetry).
    
    Args:
        game_state: The current game state
    
    Returns:
        List of encoded states [original, horizontally flipped]
    """
    original = encode_state(game_state)
    
    # Horizontal flip (left-right)
    flipped = np.flip(original, axis=2).copy()
    
    return [original, flipped]


def decode_position_planes(planes: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Extract player positions from encoded planes (for debugging).
    
    Args:
        planes: Encoded state tensor
    
    Returns:
        (player1_pos, player2_pos)
    """
    p1_pos = tuple(np.argwhere(planes[0] == 1.0)[0])
    p2_pos = tuple(np.argwhere(planes[1] == 1.0)[0])
    return p1_pos, p2_pos


def visualize_planes(planes: np.ndarray) -> str:
    """
    Create a string visualization of encoded planes (for debugging).
    
    Args:
        planes: Encoded state tensor
    
    Returns:
        Multi-line string visualization
    """
    lines = []
    plane_names = [
        "Player 1 Position",
        "Player 2 Position", 
        "Horizontal Walls",
        "Vertical Walls",
        "Current Player",
        "P1 Walls Remaining",
        "P2 Walls Remaining"
    ]
    
    for i, name in enumerate(plane_names):
        lines.append(f"\n=== Plane {i}: {name} ===")
        for row in range(GRID_COUNT):
            row_str = ""
            for col in range(GRID_COUNT):
                val = planes[i, row, col]
                if val == 0:
                    row_str += ". "
                elif val == 1:
                    row_str += "# "
                else:
                    row_str += f"{val:.1f} "
            lines.append(row_str)
    
    return "\n".join(lines)


# === Canonical form for MCTS ===

def get_canonical_state(game_state: 'GameState') -> np.ndarray:
    """
    Get canonical form of state (always from current player's perspective).
    This ensures the network always sees the game from the same viewpoint.
    
    If current player is player 2, we swap player positions and invert
    the board vertically so player 2's goal becomes row 0.
    
    Args:
        game_state: The current game state
    
    Returns:
        Canonicalized encoded state
    """
    planes = encode_state(game_state)
    
    if game_state.current_player == 2:
        # Swap player position planes
        planes[[0, 1]] = planes[[1, 0]]
        
        # Swap wall remaining planes
        planes[[5, 6]] = planes[[6, 5]]
        
        # Flip board vertically (so current player's goal is always row 0)
        planes = np.flip(planes, axis=1).copy()
        
        # Current player plane becomes 1 (since we're now from player 2's view as "player 1")
        planes[4, :, :] = 1.0
    
    return planes


def flip_action_for_player2(action_index: int) -> int:
    """
    Flip an action index when converting from player 2's canonical view
    back to the actual board.
    
    Args:
        action_index: Action index in canonical coordinates
    
    Returns:
        Action index in actual board coordinates
    """
    from AI.action_utils import index_to_action, action_to_index, GRID_COUNT
    
    action = index_to_action(action_index)
    action_type, data = action
    
    if action_type == 'move':
        row, col = data
        # Flip row (8 - row for 9x9 board)
        new_row = GRID_COUNT - 1 - row
        return action_to_index(('move', (new_row, col)))
    else:
        (row, col), orientation = data
        # Flip wall row (7 - row for 8x8 wall grid)
        new_row = GRID_COUNT - 2 - row
        return action_to_index(('wall', ((new_row, col), orientation)))
