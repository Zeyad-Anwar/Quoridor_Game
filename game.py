"""
Quoridor Game State and Logic
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional
from collections import deque
import copy

from constants import GRID_COUNT

# Type aliases
Player = Literal[1, 2]
Position = tuple[int, int]  # (row, col)
Wall = tuple[Position, Literal['H', 'V']]  # ((row, col), orientation)


@dataclass
class GameState:
    """
    Represents the complete state of a Quoridor game.
    
    Coordinate system:
    - Row 0 is the TOP of the board (Player 2's goal)
    - Row 8 is the BOTTOM of the board (Player 1's goal)
    - Player 1 starts at row 8 (bottom), tries to reach row 0
    - Player 2 starts at row 0 (top), tries to reach row 8
    
    Wall positioning:
    - Walls are placed in the gaps between tiles
    - A wall at (row, col) with orientation 'H' blocks movement 
      between rows `row` and `row+1` for columns `col` and `col+1`
    - A wall at (row, col) with orientation 'V' blocks movement
      between columns `col` and `col+1` for rows `row` and `row+1`
    - Valid wall positions: row in [0, 7], col in [0, 7]
    """
    
    # Player positions (row, col)
    player1_pos: Position = (8, 4)  # Bottom center
    player2_pos: Position = (0, 4)  # Top center
    
    # Walls remaining for each player
    player1_walls: int = 10
    player2_walls: int = 10
    
    # Set of placed walls
    walls: set[Wall] = field(default_factory=set)
    
    # Current player's turn
    current_player: Player = 1
    
    # Game over state
    winner: Optional[Player] = None
    
    def clone(self) -> GameState:
        """Create a deep copy of the game state."""
        return GameState(
            player1_pos=self.player1_pos,
            player2_pos=self.player2_pos,
            player1_walls=self.player1_walls,
            player2_walls=self.player2_walls,
            walls=set(self.walls),
            current_player=self.current_player,
            winner=self.winner
        )
    
    def get_player_pos(self, player: Player) -> Position:
        """Get position of specified player."""
        return self.player1_pos if player == 1 else self.player2_pos
    
    def set_player_pos(self, player: Player, pos: Position) -> None:
        """Set position of specified player."""
        if player == 1:
            self.player1_pos = pos
        else:
            self.player2_pos = pos
    
    def get_walls_remaining(self, player: Player) -> int:
        """Get number of walls remaining for specified player."""
        return self.player1_walls if player == 1 else self.player2_walls
    
    def use_wall(self, player: Player) -> None:
        """Decrement wall count for specified player."""
        if player == 1:
            self.player1_walls -= 1
        else:
            self.player2_walls -= 1
    
    def get_opponent(self, player: Player) -> Player:
        """Get the opponent of specified player."""
        return 2 if player == 1 else 1
    
    def is_wall_blocking(self, from_pos: Position, to_pos: Position) -> bool:
        """
        Check if there's a wall blocking movement from from_pos to to_pos.
        Assumes from_pos and to_pos are adjacent.
        """
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        # Moving vertically (up or down)
        if from_col == to_col:
            # Moving down (from_row < to_row)
            if from_row < to_row:
                wall_row = from_row
            else:  # Moving up
                wall_row = to_row
            
            # Check for horizontal walls that block this movement
            # A horizontal wall at (wall_row, c) blocks if c <= from_col <= c+1
            for c in range(max(0, from_col - 1), min(GRID_COUNT - 1, from_col + 1)):
                if ((wall_row, c), 'H') in self.walls:
                    return True
        
        # Moving horizontally (left or right)
        elif from_row == to_row:
            # Moving right (from_col < to_col)
            if from_col < to_col:
                wall_col = from_col
            else:  # Moving left
                wall_col = to_col
            
            # Check for vertical walls that block this movement
            # A vertical wall at (r, wall_col) blocks if r <= from_row <= r+1
            for r in range(max(0, from_row - 1), min(GRID_COUNT - 1, from_row + 1)):
                if ((r, wall_col), 'V') in self.walls:
                    return True
        
        return False
    
    def get_valid_moves(self, player: Player) -> list[Position]:
        """
        Get all valid movement positions for a player.
        Implements standard Quoridor movement rules including jumping and moving diagonally.
        """
        current_pos = self.get_player_pos(player)
        opponent_pos = self.get_player_pos(self.get_opponent(player))
        valid = []
        
        row, col = current_pos
        
        # Check each direction
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            new_pos = (new_row, new_col)
            
            # Check bounds
            if not (0 <= new_row < GRID_COUNT and 0 <= new_col < GRID_COUNT):
                continue
            
            # Check if wall blocks this move
            if self.is_wall_blocking(current_pos, new_pos):
                continue
            
            # Check if opponent is in the way
            if new_pos == opponent_pos:
                # Try to jump over opponent
                jump_row, jump_col = new_row + dr, new_col + dc
                jump_pos = (jump_row, jump_col)
                
                # Check if we can jump straight over
                if (0 <= jump_row < GRID_COUNT and 0 <= jump_col < GRID_COUNT 
                    and not self.is_wall_blocking(opponent_pos, jump_pos)):
                    valid.append(jump_pos)
                else:
                    # Jump is blocked or out of bounds - try diagonal jumps
                    # Get perpendicular directions
                    if dr == 0:  # Moving horizontally
                        perp_dirs = [(-1, 0), (1, 0)]
                    else:  # Moving vertically
                        perp_dirs = [(0, -1), (0, 1)]
                    
                    for pdr, pdc in perp_dirs:
                        diag_row, diag_col = new_row + pdr, new_col + pdc
                        diag_pos = (diag_row, diag_col)
                        
                        if (0 <= diag_row < GRID_COUNT and 0 <= diag_col < GRID_COUNT
                            and not self.is_wall_blocking(opponent_pos, diag_pos)):
                            valid.append(diag_pos)
            else:
                valid.append(new_pos)
        
        return valid
    
    def is_valid_wall_placement(self, wall: Wall) -> bool:
        """
        Check if a wall placement is valid.
        A wall is valid if:
        1. It's within bounds (row and col in [0, 7])
        2. Player has walls remaining
        3. It doesn't overlap with existing walls
        4. It doesn't completely block either player from reaching their goal
        """
        (row, col), orientation = wall
        
        # Check bounds (walls can only be placed in positions 0-7)
        if not (0 <= row < GRID_COUNT - 1 and 0 <= col < GRID_COUNT - 1):
            return False
        
        # Check if current player has walls
        if self.get_walls_remaining(self.current_player) <= 0:
            return False
        
        # Check for exact duplicate
        if wall in self.walls:
            return False
        
        # Check for overlapping walls
        if orientation == 'H':
            # Horizontal wall at (row, col) conflicts with:
            # - Another horizontal wall at (row, col-1) or (row, col+1)
            # - A vertical wall at (row, col) that crosses it
            if ((row, col - 1), 'H') in self.walls or ((row, col + 1), 'H') in self.walls:
                return False
            # Check crossing vertical wall
            if ((row, col), 'V') in self.walls:
                return False
        else:  # Vertical
            # Vertical wall at (row, col) conflicts with:
            # - Another vertical wall at (row-1, col) or (row+1, col)
            # - A horizontal wall at (row, col) that crosses it
            if ((row - 1, col), 'V') in self.walls or ((row + 1, col), 'V') in self.walls:
                return False
            # Check crossing horizontal wall
            if ((row, col), 'H') in self.walls:
                return False
        
        # Temporarily add wall and check if both players can still reach goals
        test_state = self.clone()
        test_state.walls.add(wall)
        
        if not test_state._can_reach_goal(1) or not test_state._can_reach_goal(2):
            return False
        
        return True
    
    def _can_reach_goal(self, player: Player) -> bool:
        """
        Check if player can reach their goal using BFS.
        Player 1 needs to reach row 0, Player 2 needs to reach row 8.
        """
        start = self.get_player_pos(player)
        goal_row = 0 if player == 1 else GRID_COUNT - 1
        
        visited = {start}
        queue = deque([start])
        
        while queue:
            current = queue.popleft()
            
            if current[0] == goal_row:
                return True
            
            # Check all adjacent positions (ignoring opponent for pathfinding)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = current[0] + dr, current[1] + dc
                new_pos = (new_row, new_col)
                
                if (0 <= new_row < GRID_COUNT and 0 <= new_col < GRID_COUNT
                    and new_pos not in visited
                    and not self.is_wall_blocking(current, new_pos)):
                    visited.add(new_pos)
                    queue.append(new_pos)
        
        return False
    
    def get_all_valid_walls(self) -> list[Wall]:
        """Get all valid wall placements for current player."""
        if self.get_walls_remaining(self.current_player) <= 0:
            return []
        
        valid_walls = []
        for row in range(GRID_COUNT - 1):
            for col in range(GRID_COUNT - 1):
                for orientation in ['H', 'V']:
                    wall = ((row, col), orientation)
                    if self.is_valid_wall_placement(wall):
                        valid_walls.append(wall)
        
        return valid_walls
    
    def get_legal_actions(self) -> list[tuple[str, Position | Wall]]:
        """
        Get all legal actions for current player.
        Returns list of ('move', position) or ('wall', wall) tuples.
        """
        actions = []
        
        # Add all valid moves
        for pos in self.get_valid_moves(self.current_player):
            actions.append(('move', pos))
        
        # Add all valid wall placements
        for wall in self.get_all_valid_walls():
            actions.append(('wall', wall))
        
        return actions
    
    def apply_action(self, action: tuple[str, Position | Wall]) -> None:
        """Apply an action to the game state. Modifies in place."""
        action_type, data = action
        
        if action_type == 'move':
            self.set_player_pos(self.current_player, data)
        elif action_type == 'wall':
            self.walls.add(data)
            self.use_wall(self.current_player)
        
        # Check for winner
        self._check_winner()
        
        # Switch turns if game not over
        if self.winner is None:
            self.current_player = self.get_opponent(self.current_player)
    
    def _check_winner(self) -> None:
        """Check if current player has won."""
        p1_row = self.player1_pos[0]
        p2_row = self.player2_pos[0]
        
        if p1_row == 0:  # Player 1 reached top
            self.winner = 1
        elif p2_row == GRID_COUNT - 1:  # Player 2 reached bottom
            self.winner = 2
    
    def is_terminal(self) -> bool:
        """Check if game is over."""
        return self.winner is not None
    
    def get_result(self, player: Player) -> float:
        """Get result from perspective of player. 1 = win, 0 = loss, 0.5 = draw."""
        if self.winner is None:
            return 0.5
        return 1.0 if self.winner == player else 0.0
    
    def __hash__(self) -> int:
        """Make GameState hashable for MCTS."""
        return hash((
            self.player1_pos,
            self.player2_pos,
            self.player1_walls,
            self.player2_walls,
            frozenset(self.walls),
            self.current_player
        ))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GameState):
            return False
        return (
            self.player1_pos == other.player1_pos
            and self.player2_pos == other.player2_pos
            and self.player1_walls == other.player1_walls
            and self.player2_walls == other.player2_walls
            and self.walls == other.walls
            and self.current_player == other.current_player
        )
