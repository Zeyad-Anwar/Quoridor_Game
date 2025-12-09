"""
Monte Carlo Tree Search (MCTS) AI for Quoridor
"""
from __future__ import annotations
import math
import random
import time
from typing import Optional

from game import GameState, Player
from constants import MCTS_TIME_LIMIT, MCTS_SIMULATIONS


class MCTSNode:
    """
    A node in the MCTS tree.
    """
    
    def __init__(
        self, 
        state: GameState, 
        parent: Optional[MCTSNode] = None,
        action: Optional[tuple] = None
    ):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this state
        
        self.children: list[MCTSNode] = []
        self.untried_actions: list[tuple] = state.get_legal_actions()
        
        self.visits = 0
        self.wins = 0.0
        
        # The player who made the move to reach this state
        # (i.e., the player whose turn it was BEFORE this state)
        self.player_just_moved = state.get_opponent(state.current_player)
    
    def ucb1(self, exploration_weight: float = 1.41) -> float:
        """
        Calculate UCB1 value for node selection.
        UCB1 = wins/visits + C * sqrt(ln(parent_visits) / visits)
        """
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.wins / self.visits
        exploration = exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        
        return exploitation + exploration
    
    def select_child(self) -> MCTSNode:
        """Select child with highest UCB1 value."""
        return max(self.children, key=lambda c: c.ucb1())
    
    def expand(self) -> MCTSNode:
        """
        Expand node by trying an untried action.
        Returns the new child node.
        """
        action = self.untried_actions.pop()
        
        # Create new state by applying action
        new_state = self.state.clone()
        new_state.apply_action(action)
        
        # Create child node
        child = MCTSNode(new_state, parent=self, action=action)
        self.children.append(child)
        
        return child
    
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried."""
        return len(self.untried_actions) == 0
    
    def is_terminal(self) -> bool:
        """Check if this node represents a terminal state."""
        return self.state.is_terminal()
    
    def update(self, result: float) -> None:
        """Update node statistics after a simulation."""
        self.visits += 1
        self.wins += result


def simulate(state: GameState) -> Player:
    """
    Run a random simulation (rollout) from the given state.
    Returns the winner of the simulation.
    """
    sim_state = state.clone()
    
    # Limit simulation depth to prevent infinite games
    max_moves = 200
    moves = 0
    
    while not sim_state.is_terminal() and moves < max_moves:
        actions = sim_state.get_legal_actions()
        
        if not actions:
            break
        
        # Use a simple heuristic: prefer moves over walls in random play
        # This speeds up simulations significantly
        move_actions = [a for a in actions if a[0] == 'move']
        
        if move_actions and random.random() < 0.8:
            action = random.choice(move_actions)
        else:
            action = random.choice(actions)
        
        sim_state.apply_action(action)
        moves += 1
    
    return sim_state.winner


def backpropagate(node: MCTSNode, winner: Optional[Player]) -> None:
    """
    Backpropagate simulation result up the tree.
    """
    while node is not None:
        node.visits += 1
        
        if winner is not None:
            # Award win to nodes where the winning player just moved
            if node.player_just_moved == winner:
                node.wins += 1.0
            else:
                # Small reward for losing to encourage exploration
                node.wins += 0.0
        else:
            # Draw - give half point
            node.wins += 0.5
        
        node = node.parent


def mcts_search(
    root_state: GameState,
    time_limit: float = MCTS_TIME_LIMIT,
    max_simulations: int = MCTS_SIMULATIONS
) -> tuple:
    """
    Perform MCTS search from the given state.
    Returns the best action found.
    """
    root = MCTSNode(root_state)
    
    start_time = time.time()
    simulations = 0
    
    while simulations < max_simulations:
        # Check time limit
        if time.time() - start_time > time_limit:
            break
        
        node = root
        
        # Selection: traverse tree using UCB1
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.select_child()
        
        # Expansion: add a new child if not terminal
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()
        
        # Simulation: random rollout
        winner = simulate(node.state)
        
        # Backpropagation: update statistics
        backpropagate(node, winner)
        
        simulations += 1
    
    # Select best action (most visited child)
    if not root.children:
        # If no children were created, return a random action
        actions = root_state.get_legal_actions()
        return random.choice(actions) if actions else None
    
    best_child = max(root.children, key=lambda c: c.visits)
    
    # Debug info
    print(f"MCTS: {simulations} simulations in {time.time() - start_time:.2f}s")
    print(f"Best action: {best_child.action} (visits: {best_child.visits}, "
          f"win rate: {best_child.wins/best_child.visits:.2%})")
    
    return best_child.action


class AIPlayer:
    """
    AI Player wrapper for easy integration with game loop.
    """
    
    def __init__(
        self, 
        player: Player,
        time_limit: float = MCTS_TIME_LIMIT,
        max_simulations: int = MCTS_SIMULATIONS
    ):
        self.player = player
        self.time_limit = time_limit
        self.max_simulations = max_simulations
    
    def get_action(self, state: GameState) -> tuple:
        """Get the best action for the current state."""
        return mcts_search(state, self.time_limit, self.max_simulations)
