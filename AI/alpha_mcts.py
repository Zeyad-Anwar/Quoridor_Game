"""
AlphaZero-style Monte Carlo Tree Search for Quoridor
Uses neural network for policy prior and value estimation instead of random rollouts.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from game import GameState
    from AI.network import QuoridorNet

from constants import (
    ALPHA_MCTS_SIMULATIONS, C_PUCT,
    DIRICHLET_ALPHA, DIRICHLET_EPSILON,
    ACTION_SPACE_SIZE
)
from AI.encoder import encode_state, get_canonical_state, flip_action_for_player2
from AI.action_utils import (
    action_to_index, index_to_action,
    get_legal_action_mask, get_legal_action_indices
)


class AlphaNode:
    """
    A node in the AlphaZero MCTS tree.
    
    Key differences from vanilla MCTS:
    - Stores prior probability P from neural network policy
    - Uses PUCT formula for selection instead of UCB1
    - No random rollouts - value comes from neural network
    """
    
    def __init__(
        self,
        state: 'GameState',
        parent: Optional['AlphaNode'] = None,
        parent_action: Optional[tuple] = None,
        prior: float = 0.0
    ):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action  # Action that led to this node
        self.prior = prior  # P(a|s) from neural network
        
        self.children: dict[int, 'AlphaNode'] = {}  # action_index -> child node
        self.visit_count = 0  # N(s, a)
        self.value_sum = 0.0  # W(s, a) - total value from this node
        self.is_expanded = False
    
    @property
    def q_value(self) -> float:
        """Mean value Q(s, a) = W(s, a) / N(s, a)"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def puct_score(self, parent_visits: int, c_puct: float = C_PUCT) -> float:
        """
        Calculate PUCT score for node selection.
        
        PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        Higher prior and fewer visits increase exploration.
        Higher Q increases exploitation.
        """
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration
    
    def select_child(self) -> 'AlphaNode':
        """Select child with highest PUCT score."""
        return max(
            self.children.values(),
            key=lambda child: child.puct_score(self.visit_count)
        )
    
    def expand(self, policy: np.ndarray) -> None:
        """
        Expand node by creating children for all legal actions.
        
        Args:
            policy: Probability distribution over actions from neural network
        """
        self.is_expanded = True
        legal_actions = self.state.get_legal_actions()
        
        for action in legal_actions:
            action_idx = action_to_index(action)
            prior = policy[action_idx]
            
            # Create child state
            child_state = self.state.clone()
            child_state.apply_action(action)
            
            # Create child node
            self.children[action_idx] = AlphaNode(
                state=child_state,
                parent=self,
                parent_action=action,
                prior=prior
            )
    
    def is_terminal(self) -> bool:
        """Check if this node represents a terminal state."""
        return self.state.is_terminal()
    
    def backpropagate(self, value: float) -> None:
        """
        Backpropagate value up the tree.
        Value is from the perspective of the player who just moved.
        """
        node = self
        # Alternate sign as we go up (opponent's win is our loss)
        current_value = value
        
        while node is not None:
            node.visit_count += 1
            node.value_sum += current_value
            current_value = -current_value  # Flip for opponent's perspective
            node = node.parent


class AlphaMCTS:
    """
    AlphaZero-style MCTS that uses a neural network for evaluation.
    """
    
    def __init__(
        self,
        network: 'QuoridorNet',
        num_simulations: int = ALPHA_MCTS_SIMULATIONS,
        c_puct: float = C_PUCT,
        dirichlet_alpha: float = DIRICHLET_ALPHA,
        dirichlet_epsilon: float = DIRICHLET_EPSILON,
        add_noise: bool = True
    ):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.add_noise = add_noise  # Add exploration noise at root
    
    def search(self, root_state: 'GameState') -> tuple[np.ndarray, float]:
        """
        Perform MCTS search from the given root state.
        
        Args:
            root_state: Starting game state
        
        Returns:
            action_probs: Visit count distribution over actions (209,)
            root_value: Estimated value of root position
        """
        root = AlphaNode(root_state)
        
        # Get initial policy and value for root
        encoded = encode_state(root_state)
        legal_mask = get_legal_action_mask(root_state)
        policy, value = self.network.predict(encoded, legal_mask)
        
        # Add Dirichlet noise at root for exploration
        if self.add_noise:
            policy = self._add_dirichlet_noise(policy, legal_mask)
        
        # Expand root
        root.expand(policy)
        root.visit_count = 1
        root.value_sum = value
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection: traverse tree using PUCT
            while node.is_expanded and not node.is_terminal():
                node = node.select_child()
                search_path.append(node)
            
            # Get value for leaf
            if node.is_terminal():
                # Terminal node: get actual game result
                winner = node.state.winner
                if winner is None:
                    leaf_value = 0.0
                else:
                    # Value from perspective of player who just moved to reach this state
                    last_player = node.state.get_opponent(node.state.current_player)
                    leaf_value = 1.0 if winner == last_player else -1.0
            else:
                # Non-terminal: expand and evaluate with network
                encoded = encode_state(node.state)
                legal_mask = get_legal_action_mask(node.state)
                policy, leaf_value = self.network.predict(encoded, legal_mask)
                
                node.expand(policy)
                # Value is from network's perspective (current player)
                # We need it from perspective of player who moved to reach this node
                leaf_value = -leaf_value  # Flip for parent's perspective
            
            # Backpropagation
            node.backpropagate(leaf_value)
        
        # Extract visit counts as action probabilities
        action_probs = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        for action_idx, child in root.children.items():
            action_probs[action_idx] = child.visit_count
        
        # Normalize to probability distribution
        total_visits = np.sum(action_probs)
        if total_visits > 0:
            action_probs = action_probs / total_visits
        
        # Root value estimate
        root_value = root.q_value
        
        return action_probs, root_value
    
    def _add_dirichlet_noise(
        self, 
        policy: np.ndarray, 
        legal_mask: np.ndarray
    ) -> np.ndarray:
        """
        Add Dirichlet noise to policy for exploration at root.
        
        noise_policy = (1 - ε) * policy + ε * noise
        """
        legal_indices = np.where(legal_mask)[0]
        num_legal = len(legal_indices)
        
        if num_legal == 0:
            return policy
        
        # Generate Dirichlet noise only for legal actions
        noise = np.random.dirichlet([self.dirichlet_alpha] * num_legal)
        
        noisy_policy = policy.copy()
        for i, idx in enumerate(legal_indices):
            noisy_policy[idx] = (
                (1 - self.dirichlet_epsilon) * policy[idx] +
                self.dirichlet_epsilon * noise[i]
            )
        
        return noisy_policy
    
    def get_action(
        self, 
        state: 'GameState', 
        temperature: float = 1.0
    ) -> tuple:
        """
        Get best action for the given state.
        
        Args:
            state: Current game state
            temperature: Controls exploration
                - temp=0: Always pick most visited action
                - temp=1: Sample proportional to visit counts
                - temp>1: More random
        
        Returns:
            Selected action tuple
        """
        action_probs, _ = self.search(state)
        
        if temperature == 0:
            # Greedy: pick most visited
            action_idx = np.argmax(action_probs)
        else:
            # Sample with temperature
            temp_probs = np.power(action_probs, 1.0 / temperature)
            temp_probs = temp_probs / np.sum(temp_probs)
            action_idx = np.random.choice(len(temp_probs), p=temp_probs)
        
        return index_to_action(action_idx)
    
    def get_action_probs(
        self, 
        state: 'GameState', 
        temperature: float = 1.0
    ) -> tuple[np.ndarray, float]:
        """
        Get action probability distribution for training.
        
        Args:
            state: Current game state
            temperature: Controls distribution sharpness
        
        Returns:
            action_probs: Probability distribution over actions
            value: Estimated position value
        """
        action_probs, value = self.search(state)
        
        if temperature != 1.0 and temperature > 0:
            # Apply temperature
            temp_probs = np.power(action_probs, 1.0 / temperature)
            total = np.sum(temp_probs)
            if total > 0:
                action_probs = temp_probs / total
        
        return action_probs, value


class AlphaZeroPlayer:
    """
    AI Player wrapper using AlphaZero MCTS.
    Drop-in replacement for the original AIPlayer class.
    """
    
    def __init__(
        self,
        player: int,
        network: 'QuoridorNet',
        num_simulations: int = ALPHA_MCTS_SIMULATIONS,
        temperature: float = 0.0  # Default to greedy for playing
    ):
        self.player = player
        self.network = network
        self.mcts = AlphaMCTS(
            network=network,
            num_simulations=num_simulations,
            add_noise=False  # No noise when playing (only for training)
        )
        self.temperature = temperature
    
    def get_action(self, state: 'GameState') -> tuple:
        """Get the best action for the current state."""
        return self.mcts.get_action(state, temperature=self.temperature)
