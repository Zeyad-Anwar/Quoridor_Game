"""
Action Encoding/Decoding for AlphaZero
Maps between game actions and neural network indices.

Action Space (209 total):
- Indices 0-80: Move actions (row * 9 + col) -> position on 9x9 board
- Indices 81-144: Horizontal wall placements (row * 8 + col) on 8x8 grid
- Indices 145-208: Vertical wall placements (row * 8 + col) on 8x8 grid
"""
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game import GameState, Position, Wall

from constants import GRID_COUNT, ACTION_SPACE_SIZE


# Index boundaries
MOVE_START = 0
MOVE_END = 81  # 9 * 9
HWALL_START = 81
HWALL_END = 145  # 81 + 8 * 8
VWALL_START = 145
VWALL_END = 209  # 145 + 8 * 8


def action_to_index(action: tuple[str, any]) -> int:
    """
    Convert a game action to neural network index.
    
    Args:
        action: ('move', (row, col)) or ('wall', ((row, col), 'H'/'V'))
    
    Returns:
        Index in range [0, 208]
    """
    action_type, data = action
    
    if action_type == 'move':
        row, col = data
        return MOVE_START + row * GRID_COUNT + col
    
    elif action_type == 'wall':
        (row, col), orientation = data
        if orientation == 'H':
            return HWALL_START + row * (GRID_COUNT - 1) + col
        else:  # 'V'
            return VWALL_START + row * (GRID_COUNT - 1) + col
    
    raise ValueError(f"Unknown action type: {action_type}")


def index_to_action(index: int) -> tuple[str, any]:
    """
    Convert neural network index to game action.
    
    Args:
        index: Index in range [0, 208]
    
    Returns:
        ('move', (row, col)) or ('wall', ((row, col), 'H'/'V'))
    """
    if index < 0 or index >= ACTION_SPACE_SIZE:
        raise ValueError(f"Index {index} out of range [0, {ACTION_SPACE_SIZE})")
    
    if index < HWALL_START:
        # Move action
        row = index // GRID_COUNT
        col = index % GRID_COUNT
        return ('move', (row, col))
    
    elif index < VWALL_START:
        # Horizontal wall
        wall_index = index - HWALL_START
        row = wall_index // (GRID_COUNT - 1)
        col = wall_index % (GRID_COUNT - 1)
        return ('wall', ((row, col), 'H'))
    
    else:
        # Vertical wall
        wall_index = index - VWALL_START
        row = wall_index // (GRID_COUNT - 1)
        col = wall_index % (GRID_COUNT - 1)
        return ('wall', ((row, col), 'V'))


def get_legal_action_mask(game_state: 'GameState') -> np.ndarray:
    """
    Get a boolean mask of legal actions for the current state.
    
    Args:
        game_state: Current game state
    
    Returns:
        Boolean array of shape (ACTION_SPACE_SIZE,) where True = legal
    """
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
    
    # Get legal actions from game state (uses cache if available)
    legal_actions = game_state.get_legal_actions()
    
    for action in legal_actions:
        idx = action_to_index(action)
        mask[idx] = True
    
    return mask


def get_legal_action_mask_and_actions(game_state: 'GameState') -> tuple[np.ndarray, list]:
    """
    Get both the legal action mask and the list of legal actions.
    More efficient than calling both functions separately.
    
    Args:
        game_state: Current game state
    
    Returns:
        Tuple of (mask, legal_actions)
    """
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
    legal_actions = game_state.get_legal_actions()
    
    for action in legal_actions:
        idx = action_to_index(action)
        mask[idx] = True
    
    return mask, legal_actions


def get_legal_action_indices(game_state: 'GameState') -> list[int]:
    """
    Get list of indices for legal actions.
    
    Args:
        game_state: Current game state
    
    Returns:
        List of legal action indices
    """
    legal_actions = game_state.get_legal_actions()
    return [action_to_index(action) for action in legal_actions]


def mask_illegal_actions(logits: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """
    Mask illegal actions by setting their logits to -infinity.
    
    Args:
        logits: Raw logits from policy network, shape (ACTION_SPACE_SIZE,)
        legal_mask: Boolean mask of legal actions
    
    Returns:
        Masked logits with illegal actions set to -inf
    """
    masked = logits.copy()
    masked[~legal_mask] = -np.inf
    return masked


def softmax_with_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Apply softmax with temperature to logits.
    
    Args:
        logits: Input logits (may contain -inf for masked actions)
        temperature: Temperature parameter (higher = more random)
    
    Returns:
        Probability distribution over actions
    """
    if temperature <= 0:
        # Greedy selection
        probs = np.zeros_like(logits)
        probs[np.argmax(logits)] = 1.0
        return probs
    
    # Scale by temperature
    scaled = logits / temperature
    
    # Numerical stability: subtract max
    scaled = scaled - np.max(scaled[np.isfinite(scaled)])
    
    # Compute softmax (exp(-inf) = 0)
    exp_scaled = np.exp(scaled)
    exp_scaled[~np.isfinite(logits)] = 0  # Ensure masked stay 0
    
    total = np.sum(exp_scaled)
    if total == 0:
        # Fallback: uniform over legal actions
        legal_count = np.sum(np.isfinite(logits))
        if legal_count > 0:
            probs = np.where(np.isfinite(logits), 1.0 / legal_count, 0.0)
            return probs
        return exp_scaled
    
    return exp_scaled / total


# === Utility functions for debugging ===

def action_to_string(action: tuple[str, any]) -> str:
    """Convert action to human-readable string."""
    action_type, data = action
    
    if action_type == 'move':
        row, col = data
        return f"Move to ({row}, {col})"
    else:
        (row, col), orientation = data
        orient_str = "Horizontal" if orientation == 'H' else "Vertical"
        return f"{orient_str} wall at ({row}, {col})"


def print_action_distribution(probs: np.ndarray, top_k: int = 5) -> None:
    """Print top-k actions by probability."""
    indices = np.argsort(probs)[::-1][:top_k]
    
    print(f"Top {top_k} actions:")
    for i, idx in enumerate(indices):
        action = index_to_action(idx)
        print(f"  {i+1}. {action_to_string(action)}: {probs[idx]:.4f}")
