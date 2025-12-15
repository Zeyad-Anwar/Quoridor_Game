"""
AlphaZero-style Monte Carlo Tree Search for Quoridor
Uses neural network for policy prior and value estimation instead of random rollouts.

Performance Optimizations:
- Batched neural network inference (batch_size configurable)
- Virtual loss for parallel tree traversal
- Lazy child node creation
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
    ACTION_SPACE_SIZE, MCTS_BATCH_SIZE
)
from AI.encoder import encode_state
from AI.action_utils import (
    action_to_index, index_to_action,
    get_legal_action_mask, get_legal_action_indices,
    get_legal_action_mask_and_actions
)

# Virtual loss value - penalizes nodes being evaluated to encourage exploration
VIRTUAL_LOSS = 3.0


class AlphaNode:
    """
    A node in the AlphaZero MCTS tree.
    
    Key differences from vanilla MCTS:
    - Stores prior probability P from neural network policy
    - Uses PUCT formula for selection instead of UCB1
    - No random rollouts - value comes from neural network
    - LAZY EXPANSION: Child states are only created when actually visited
    - Virtual loss support for parallel/batched MCTS
    """
    
    __slots__ = ('state', 'parent', 'parent_action', 'prior', 'children', 
                 'visit_count', 'value_sum', 'is_expanded', '_legal_actions', 
                 '_policy', 'virtual_loss')
    
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
        self._legal_actions: Optional[list] = None  # Cached legal actions
        self._policy: Optional[np.ndarray] = None  # Policy for lazy child creation
        self.virtual_loss = 0  # Virtual loss for parallel MCTS
    
    @property
    def q_value(self) -> float:
        """Mean value Q(s, a) = W(s, a) / N(s, a), accounting for virtual loss."""
        effective_visits = self.visit_count + self.virtual_loss
        if effective_visits == 0:
            return 0.0
        # Virtual loss acts as if we got 'virtual_loss' number of losses
        effective_value = self.value_sum - self.virtual_loss * VIRTUAL_LOSS
        return effective_value / effective_visits
    
    def puct_score(self, parent_visits: int, c_puct: float = C_PUCT) -> float:
        """
        Calculate PUCT score for node selection.
        
        PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        Higher prior and fewer visits increase exploration.
        Higher Q increases exploitation.
        Virtual loss reduces score for nodes being evaluated.
        """
        effective_visits = self.visit_count + self.virtual_loss
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + effective_visits)
        return self.q_value + exploration
    
    def apply_virtual_loss(self) -> None:
        """Apply virtual loss to this node and all ancestors."""
        node = self
        while node is not None:
            node.virtual_loss += 1
            node = node.parent
    
    def revert_virtual_loss(self) -> None:
        """Revert virtual loss from this node and all ancestors."""
        node = self
        while node is not None:
            node.virtual_loss -= 1
            node = node.parent
    
    def _get_or_create_child(self, action_idx: int) -> 'AlphaNode':
        """Lazily create child node only when needed."""
        if action_idx in self.children:
            return self.children[action_idx]
        
        # Create child on demand
        action = index_to_action(action_idx)
        child_state = self.state.clone()
        child_state.apply_action(action)
        
        prior = self._policy[action_idx] if self._policy is not None else 0.0
        child = AlphaNode(
            state=child_state,
            parent=self,
            parent_action=action,
            prior=prior
        )
        self.children[action_idx] = child
        return child
    
    def select_child(self) -> 'AlphaNode':
        """Select child with highest PUCT score (lazy creation)."""
        if self._legal_actions is None:
            return max(
                self.children.values(),
                key=lambda child: child.puct_score(self.visit_count)
            )
        
        # For lazy expansion: calculate PUCT for all legal actions
        best_score = -float('inf')
        best_action_idx = -1
        
        sqrt_parent = math.sqrt(self.visit_count)
        
        for action in self._legal_actions:
            action_idx = action_to_index(action)
            
            if action_idx in self.children:
                child = self.children[action_idx]
                score = child.puct_score(self.visit_count)
            else:
                # Unvisited child: Q=0, N=0
                prior = self._policy[action_idx] if self._policy is not None else 0.0
                score = C_PUCT * prior * sqrt_parent  # (1 + 0) = 1
            
            if score > best_score:
                best_score = score
                best_action_idx = action_idx
        
        return self._get_or_create_child(best_action_idx)
    
    def expand(self, policy: np.ndarray, legal_actions: list = None) -> None:
        """
        Mark node as expanded and store policy for lazy child creation.
        
        Args:
            policy: Probability distribution over actions from neural network
            legal_actions: Optional pre-computed legal actions (avoids recomputation)
        """
        self.is_expanded = True
        self._policy = policy
        self._legal_actions = legal_actions if legal_actions is not None else self.state.get_legal_actions()
        # Don't create children here - they'll be created lazily in select_child
    
    def is_terminal(self) -> bool:
        """Check if this node represents a terminal state."""
        return self.state.is_terminal()
    
    def backpropagate(self, value: float) -> None:
        """
        Backpropagate value up the tree.
        Value is from the perspective of the CURRENT player at this node.
        We store it, then flip when moving to parent (opponent's perspective).
        """
        node = self
        current_value = value
        
        while node is not None:
            node.visit_count += 1
            node.value_sum += current_value
            # Flip AFTER storing, for the parent's perspective
            node = node.parent
            current_value = -current_value


class AlphaMCTS:
    """
    AlphaZero-style MCTS that uses a neural network for evaluation.
    
    Performance optimizations:
    - Batched neural network inference
    - Virtual loss for parallel tree traversal
    """
    
    def __init__(
        self,
        network: 'QuoridorNet',
        num_simulations: int = ALPHA_MCTS_SIMULATIONS,
        c_puct: float = C_PUCT,
        dirichlet_alpha: float = DIRICHLET_ALPHA,
        dirichlet_epsilon: float = DIRICHLET_EPSILON,
        add_noise: bool = True,
        batch_size: int = MCTS_BATCH_SIZE
    ):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.add_noise = add_noise  # Add exploration noise at root
        self.batch_size = batch_size
    
    def search(self, root_state: 'GameState') -> tuple[np.ndarray, float]:
        """
        Perform MCTS search from the given root state with batched inference.
        
        Args:
            root_state: Starting game state
        
        Returns:
            action_probs: Visit count distribution over actions (209,)
            root_value: Estimated value of root position
        """
        root = AlphaNode(root_state)
        
        # Get initial policy and value for root
        # Note: We use encode_state (not canonical) because legal actions are in board space
        encoded = encode_state(root_state)
        legal_mask, legal_actions = get_legal_action_mask_and_actions(root_state)
        policy, value = self.network.predict(encoded, legal_mask)
        
        # Add Dirichlet noise at root for exploration
        if self.add_noise:
            policy = self._add_dirichlet_noise(policy, legal_mask)
        
        # Expand root with pre-computed legal actions
        root.expand(policy, legal_actions)
        root.visit_count = 1
        root.value_sum = value
        
        # Run simulations in batches
        simulations_done = 0
        while simulations_done < self.num_simulations:
            # Collect a batch of leaf nodes to evaluate
            current_batch_size = min(self.batch_size, self.num_simulations - simulations_done)
            leaves_to_eval = []  # (node, encoded_state, legal_mask, legal_actions)
            terminal_leaves = []  # (node, value)
            expanded_leaves = []  # nodes that were already expanded (just backprop)
            
            for _ in range(current_batch_size):
                node = root
                
                # Selection: traverse tree using PUCT
                while node.is_expanded and not node.is_terminal():
                    node = node.select_child()
                
                # Apply virtual loss to discourage other paths from selecting same node
                node.apply_virtual_loss()
                
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
                    terminal_leaves.append((node, leaf_value))
                elif node.is_expanded:
                    # Node was already expanded (e.g., by another path in this batch)
                    # Just use its existing value estimate
                    expanded_leaves.append(node)
                else:
                    # Non-terminal, unexpanded: need to evaluate with network
                    # Note: We use encode_state (not canonical) because legal actions are in board space
                    encoded = encode_state(node.state)
                    legal_mask_node, legal_actions_node = get_legal_action_mask_and_actions(node.state)
                    leaves_to_eval.append((node, encoded, legal_mask_node, legal_actions_node))
            
            # Batch evaluate non-terminal leaves
            if leaves_to_eval:
                # Stack encoded states for batch inference
                states_batch = np.stack([item[1] for item in leaves_to_eval], axis=0)
                masks_batch = np.stack([item[2] for item in leaves_to_eval], axis=0)
                
                # Batch predict
                policies, values = self.network.predict_batch(states_batch, masks_batch)
                
                # Expand nodes and backpropagate
                for i, (node, encoded, legal_mask_node, legal_actions_node) in enumerate(leaves_to_eval):
                    policy_i = policies[i]
                    leaf_value = float(values[i])
                    
                    node.expand(policy_i, legal_actions_node)
                    # Value is from network's perspective (current player at this node)
                    # backpropagate expects value from current player's perspective
                    # so we pass it directly without flipping
                    
                    # Revert virtual loss before backpropagation
                    node.revert_virtual_loss()
                    node.backpropagate(leaf_value)
            
            # Handle terminal leaves
            for node, leaf_value in terminal_leaves:
                node.revert_virtual_loss()
                # leaf_value is from the perspective of the player who just moved
                # but current_player at terminal node is the opponent
                # so we need to flip to get value from current player's perspective
                node.backpropagate(-leaf_value)
            
            # Handle already-expanded leaves (use their q_value)
            for node in expanded_leaves:
                node.revert_virtual_loss()
                # q_value is already from current player's perspective at this node
                node.backpropagate(node.q_value)
            
            simulations_done += current_batch_size
        
        # Extract visit counts as action probabilities
        action_probs = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        for action_idx, child in root.children.items():
            action_probs[action_idx] = child.visit_count
        
        # Normalize to probability distribution
        total_visits = np.sum(action_probs)
        if total_visits > 0:
            action_probs = action_probs / total_visits
        
        # Root value estimate - use real q_value without virtual loss effect
        root_value = root.value_sum / root.visit_count if root.visit_count > 0 else 0.0
        
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
