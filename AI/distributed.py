"""
Ray-based Distributed Training for AlphaZero Quoridor

Architecture:
- CPU Machine (Head Node): Runs Ray head, self-play workers, replay buffer
- GPU Machine (Worker Node): Runs inference server and training actor

Usage:
    # On CPU machine (head):
    ray start --head --port=6379
    
    # On GPU machine:
    ray start --address='<cpu-machine-ip>:6379' --num-gpus=1
    
    # Then run training from CPU machine:
    python -m AI.distributed
"""
from __future__ import annotations
import os
import sys
import time
import random
import signal
import pickle
from dataclasses import dataclass
from typing import Optional, Any
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray
from ray.util.queue import Queue as RayQueue

from game import GameState
from AI.network import QuoridorNet, create_network
from AI.encoder import encode_state
from AI.action_utils import (
    action_to_index, index_to_action,
    get_legal_action_mask, get_legal_action_mask_and_actions
)
from constants import (
    SELF_PLAY_GAMES, REPLAY_BUFFER_SIZE, BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY, TRAINING_ITERATIONS,
    CHECKPOINT_INTERVAL, TEMP_THRESHOLD, TEMP_INIT, TEMP_FINAL,
    ALPHA_MCTS_SIMULATIONS, ACTION_SPACE_SIZE, GRID_COUNT,
    NUM_PARALLEL_GAMES, AUTO_SAVE_MINUTES, EVAL_INTERVAL, EVAL_GAMES,
    WIN_RATE_THRESHOLD, BOOTSTRAP_TEMP_INIT, BOOTSTRAP_TEMP_FINAL, 
    BOOTSTRAP_TEMP_THRESHOLD, C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON,
    RAY_LOCAL_MODE, RAY_HEAD_ADDRESS, RAY_HEAD_PORT,
    RAY_INFERENCE_BATCH_SIZE, RAY_WEIGHT_SYNC_INTERVAL, RAY_NUM_WORKERS,
    RAY_OBJECT_STORE_MEMORY, INPUT_CHANNELS
)

# Try to import TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

# Try newer torch.amp API
try:
    from torch.amp import autocast, GradScaler
    AMP_DEVICE_TYPE = 'cuda'
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_DEVICE_TYPE = None


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TrainingExample:
    """A single training example from self-play."""
    state: np.ndarray          # Encoded board state (8, 9, 9)
    policy: np.ndarray         # MCTS visit count distribution (209,)
    value: float               # Game outcome [-1, 1]


class TrainingDataset(Dataset):
    """PyTorch Dataset wrapper for training examples."""
    
    def __init__(self, examples: list[TrainingExample]):
        self.examples = examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ex = self.examples[idx]
        state = torch.FloatTensor(ex.state)
        policy = torch.FloatTensor(ex.policy)
        value = torch.FloatTensor([ex.value])
        return state, policy, value


# =============================================================================
# Ray Actors
# =============================================================================

@ray.remote(num_gpus=1)
class InferenceServer:
    """
    GPU-based inference server for neural network predictions.
    Runs on the GPU machine and handles all neural network inference.
    """
    
    def __init__(self, network_weights: Optional[dict] = None):
        """Initialize the inference server with optional pre-trained weights."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[InferenceServer] Initializing on device: {self.device}")
        
        self.network = create_network(self.device)
        
        if network_weights is not None:
            self.network.load_state_dict(network_weights)
            print("[InferenceServer] Loaded provided weights")
        
        self.network.eval()
        self.request_count = 0
    
    def predict_batch(
        self, 
        states: np.ndarray, 
        legal_masks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Batch inference for MCTS leaf evaluation.
        
        Args:
            states: Batch of encoded states (batch, 8, 9, 9)
            legal_masks: Batch of legal action masks (batch, 209)
        
        Returns:
            policies: Batch of action probabilities (batch, 209)
            values: Batch of position evaluations (batch,)
        """
        self.request_count += 1
        
        with torch.inference_mode():
            x = torch.from_numpy(states.astype(np.float32)).to(self.device, non_blocking=True)
            
            policy_logits, values = self.network(x)
            
            # Apply legal mask
            mask_tensor = torch.from_numpy(legal_masks).to(self.device, non_blocking=True)
            policy_logits = policy_logits.masked_fill(~mask_tensor, float('-inf'))
            
            # Softmax on GPU
            policies = torch.softmax(policy_logits, dim=1)
            
            return policies.cpu().numpy(), values.cpu().numpy().flatten()
    
    def predict_single(
        self, 
        state: np.ndarray, 
        legal_mask: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Single state inference."""
        policies, values = self.predict_batch(
            state[np.newaxis, ...], 
            legal_mask[np.newaxis, ...]
        )
        return policies[0], float(values[0])
    
    def get_weights(self) -> dict:
        """Get current network weights."""
        return {k: v.cpu() for k, v in self.network.state_dict().items()}
    
    def update_weights(self, weights: dict) -> None:
        """Update network weights."""
        self.network.load_state_dict(weights)
        self.network.eval()
        print(f"[InferenceServer] Weights updated (total requests: {self.request_count})")
    
    def get_stats(self) -> dict:
        """Get server statistics."""
        return {
            'request_count': self.request_count,
            'device': self.device
        }


@ray.remote(num_gpus=1)
class TrainingActor:
    """
    GPU-based training actor.
    Handles gradient updates and maintains the authoritative model weights.
    """
    
    def __init__(
        self, 
        network_weights: Optional[dict] = None,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "runs"
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_amp = self.device == 'cuda'
        print(f"[TrainingActor] Initializing on device: {self.device}")
        
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize network
        self.network = create_network(self.device)
        if network_weights is not None:
            self.network.load_state_dict(network_weights)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # LR Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=TRAINING_ITERATIONS,
            eta_min=LEARNING_RATE / 100
        )
        
        # Mixed precision
        if self.use_amp:
            if AMP_DEVICE_TYPE:
                self.scaler = GradScaler(AMP_DEVICE_TYPE)
            else:
                self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # TensorBoard
        self.writer = None
        if HAS_TENSORBOARD:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.writer = SummaryWriter(os.path.join(log_dir, f"distributed_{timestamp}"))
        
        # Training state
        self.current_iteration = 0
        self.best_win_rate = 0.0
        self.total_examples_trained = 0
    
    def train_on_examples(
        self, 
        examples: list[TrainingExample], 
        epochs: int = 10
    ) -> dict:
        """
        Train on a batch of examples.
        
        Args:
            examples: List of training examples
            epochs: Number of training epochs
        
        Returns:
            Dictionary of training metrics
        """
        self.network.train()
        
        dataset = TrainingDataset(examples)
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device == 'cuda'
        )
        
        metrics = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': []
        }
        
        for epoch in range(epochs):
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_total_loss = 0.0
            num_batches = 0
            
            for states, policies, values in dataloader:
                states = states.to(self.device)
                policies = policies.to(self.device)
                values = values.to(self.device)
                
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    if AMP_DEVICE_TYPE:
                        with autocast(AMP_DEVICE_TYPE):
                            policy_logits, pred_values = self.network(states)
                            policy_loss = -torch.mean(
                                torch.sum(policies * torch.log_softmax(policy_logits, dim=1), dim=1)
                            )
                            value_loss = nn.functional.mse_loss(pred_values, values)
                            total_loss = policy_loss + 2.0 * value_loss
                    else:
                        with autocast():
                            policy_logits, pred_values = self.network(states)
                            policy_loss = -torch.mean(
                                torch.sum(policies * torch.log_softmax(policy_logits, dim=1), dim=1)
                            )
                            value_loss = nn.functional.mse_loss(pred_values, values)
                            total_loss = policy_loss + 2.0 * value_loss
                    
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    policy_logits, pred_values = self.network(states)
                    policy_loss = -torch.mean(
                        torch.sum(policies * torch.log_softmax(policy_logits, dim=1), dim=1)
                    )
                    value_loss = nn.functional.mse_loss(pred_values, values)
                    total_loss = policy_loss + 2.0 * value_loss
                    
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_total_loss += total_loss.item()
                num_batches += 1
            
            if num_batches > 0:
                metrics['policy_loss'].append(epoch_policy_loss / num_batches)
                metrics['value_loss'].append(epoch_value_loss / num_batches)
                metrics['total_loss'].append(epoch_total_loss / num_batches)
        
        self.total_examples_trained += len(examples)
        self.network.eval()
        
        return metrics
    
    def step_scheduler(self) -> float:
        """Step the learning rate scheduler."""
        self.scheduler.step()
        return self.optimizer.param_groups[0]['lr']
    
    def get_weights(self) -> dict:
        """Get current network weights."""
        return {k: v.cpu() for k, v in self.network.state_dict().items()}
    
    def log_metrics(self, metrics: dict, iteration: int) -> None:
        """Log metrics to TensorBoard."""
        if self.writer:
            for key, value in metrics.items():
                if isinstance(value, list):
                    value = np.mean(value)
                self.writer.add_scalar(f'Train/{key}', value, iteration)
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar to TensorBoard."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def save_checkpoint(self, filepath: str, extra_state: dict = None) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'iteration': self.current_iteration,
            'state_dict': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_win_rate': self.best_win_rate,
            'total_examples_trained': self.total_examples_trained,
            'timestamp': datetime.now().isoformat(),
        }
        if extra_state:
            checkpoint.update(extra_state)
        
        torch.save(checkpoint, filepath)
        print(f"[TrainingActor] Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> dict:
        """Load training checkpoint and return extra state."""
        if not os.path.exists(filepath):
            return {}
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        try:
            self.network.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            if 'scheduler_state' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        except RuntimeError as e:
            print(f"[TrainingActor] Warning: Could not load checkpoint: {e}")
            return {}
        
        self.current_iteration = checkpoint.get('iteration', 0)
        self.best_win_rate = checkpoint.get('best_win_rate', 0.0)
        self.total_examples_trained = checkpoint.get('total_examples_trained', 0)
        
        print(f"[TrainingActor] Loaded checkpoint from iteration {self.current_iteration}")
        return checkpoint
    
    def set_iteration(self, iteration: int) -> None:
        """Set current iteration."""
        self.current_iteration = iteration
    
    def set_best_win_rate(self, win_rate: float) -> None:
        """Set best win rate."""
        self.best_win_rate = win_rate
    
    def get_state(self) -> dict:
        """Get current training state."""
        return {
            'iteration': self.current_iteration,
            'best_win_rate': self.best_win_rate,
            'total_examples_trained': self.total_examples_trained,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.writer:
            self.writer.close()


@ray.remote
class ReplayBufferActor:
    """
    Shared replay buffer accessible by all workers.
    Stores training examples from self-play.
    """
    
    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE):
        from collections import deque
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.total_added = 0
    
    def add_examples(self, examples: list[TrainingExample]) -> int:
        """Add multiple training examples."""
        for ex in examples:
            self.buffer.append(ex)
        self.total_added += len(examples)
        return len(self.buffer)
    
    def sample(self, batch_size: int) -> list[TrainingExample]:
        """Sample a random batch of examples."""
        sample_size = min(batch_size, len(self.buffer))
        return random.sample(list(self.buffer), sample_size)
    
    def get_all(self) -> list[TrainingExample]:
        """Get all examples in buffer."""
        return list(self.buffer)
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
    
    def save(self, filepath: str) -> None:
        """Save buffer to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.buffer), f)
    
    def load(self, filepath: str) -> int:
        """Load buffer from file."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                examples = pickle.load(f)
                self.buffer = type(self.buffer)(examples, maxlen=self.capacity)
                return len(self.buffer)
        return 0
    
    def get_stats(self) -> dict:
        """Get buffer statistics."""
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'total_added': self.total_added
        }


# =============================================================================
# Distributed MCTS
# =============================================================================

class DistributedAlphaMCTS:
    """
    MCTS that uses remote inference server for neural network predictions.
    Batches leaf nodes and sends them to the GPU for evaluation.
    """
    
    def __init__(
        self,
        inference_server,  # Ray actor handle
        num_simulations: int = ALPHA_MCTS_SIMULATIONS,
        c_puct: float = C_PUCT,
        dirichlet_alpha: float = DIRICHLET_ALPHA,
        dirichlet_epsilon: float = DIRICHLET_EPSILON,
        add_noise: bool = True,
        batch_size: int = RAY_INFERENCE_BATCH_SIZE
    ):
        self.inference_server = inference_server
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.add_noise = add_noise
        self.batch_size = batch_size
    
    def search(self, root_state: 'GameState') -> tuple[np.ndarray, float]:
        """
        Perform MCTS search with remote inference.
        """
        from AI.alpha_mcts import AlphaNode  # Import here to avoid circular
        
        root = AlphaNode(root_state)
        
        # Get initial policy for root (single inference)
        encoded = encode_state(root_state)
        legal_mask, legal_actions = get_legal_action_mask_and_actions(root_state)
        
        policy, value = ray.get(
            self.inference_server.predict_single.remote(encoded, legal_mask)
        )
        
        # Add Dirichlet noise at root
        if self.add_noise:
            policy = self._add_dirichlet_noise(policy, legal_mask)
        
        root.expand(policy, legal_actions)
        root.visit_count = 1
        root.value_sum = value
        
        # Run simulations in batches
        simulations_done = 0
        while simulations_done < self.num_simulations:
            current_batch_size = min(self.batch_size, self.num_simulations - simulations_done)
            leaves_to_eval = []
            terminal_leaves = []
            expanded_leaves = []
            
            for _ in range(current_batch_size):
                node = root
                
                while node.is_expanded and not node.is_terminal():
                    node = node.select_child()
                
                node.apply_virtual_loss()
                
                if node.is_terminal():
                    winner = node.state.winner
                    if winner is None:
                        leaf_value = 0.0
                    else:
                        last_player = node.state.get_opponent(node.state.current_player)
                        leaf_value = 1.0 if winner == last_player else -1.0
                    terminal_leaves.append((node, leaf_value))
                elif node.is_expanded:
                    expanded_leaves.append(node)
                else:
                    encoded = encode_state(node.state)
                    legal_mask_node, legal_actions_node = get_legal_action_mask_and_actions(node.state)
                    leaves_to_eval.append((node, encoded, legal_mask_node, legal_actions_node))
            
            # Batch evaluate via remote inference
            if leaves_to_eval:
                states_batch = np.stack([item[1] for item in leaves_to_eval], axis=0)
                masks_batch = np.stack([item[2] for item in leaves_to_eval], axis=0)
                
                # Remote call to GPU
                policies, values = ray.get(
                    self.inference_server.predict_batch.remote(states_batch, masks_batch)
                )
                
                for i, (node, encoded, legal_mask_node, legal_actions_node) in enumerate(leaves_to_eval):
                    node.expand(policies[i], legal_actions_node)
                    node.revert_virtual_loss()
                    node.backpropagate(float(values[i]))
            
            for node, leaf_value in terminal_leaves:
                node.revert_virtual_loss()
                node.backpropagate(-leaf_value)
            
            for node in expanded_leaves:
                node.revert_virtual_loss()
                node.backpropagate(node.q_value)
            
            simulations_done += current_batch_size
        
        # Extract action probabilities
        action_probs = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        for action_idx, child in root.children.items():
            action_probs[action_idx] = child.visit_count
        
        total_visits = np.sum(action_probs)
        if total_visits > 0:
            action_probs = action_probs / total_visits
        
        root_value = root.value_sum / root.visit_count if root.visit_count > 0 else 0.0
        
        return action_probs, root_value
    
    def _add_dirichlet_noise(self, policy: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
        """Add Dirichlet noise for exploration."""
        legal_indices = np.where(legal_mask)[0]
        num_legal = len(legal_indices)
        
        if num_legal == 0:
            return policy
        
        noise = np.random.dirichlet([self.dirichlet_alpha] * num_legal)
        noisy_policy = policy.copy()
        
        for i, idx in enumerate(legal_indices):
            noisy_policy[idx] = (
                (1 - self.dirichlet_epsilon) * policy[idx] +
                self.dirichlet_epsilon * noise[i]
            )
        
        return noisy_policy
    
    def get_action_probs(
        self, 
        state: 'GameState', 
        temperature: float = 1.0
    ) -> tuple[np.ndarray, float]:
        """Get action probability distribution."""
        action_probs, value = self.search(state)
        
        if temperature != 1.0 and temperature > 0:
            temp_probs = np.power(action_probs, 1.0 / temperature)
            total = np.sum(temp_probs)
            if total > 0:
                action_probs = temp_probs / total
        
        return action_probs, value


# =============================================================================
# Self-Play Task
# =============================================================================

@ray.remote
def self_play_game_distributed(
    inference_server,
    num_simulations: int = ALPHA_MCTS_SIMULATIONS,
    temp_threshold: int = TEMP_THRESHOLD,
    is_bootstrap: bool = False
) -> tuple[list[dict], int]:
    """
    Play one self-play game using distributed inference.
    
    Returns:
        Tuple of (training examples as dicts, number of moves)
    """
    mcts = DistributedAlphaMCTS(
        inference_server=inference_server,
        num_simulations=num_simulations,
        add_noise=True
    )
    
    game_state = GameState()
    game_history = []
    move_count = 0
    max_moves = 200
    
    # For bootstrap: network plays as player 2, random as player 1
    network_player = 2 if is_bootstrap else None
    
    while not game_state.is_terminal() and move_count < max_moves:
        current = game_state.current_player
        
        # Bootstrap mode: random opponent for player 1
        if is_bootstrap and current != network_player:
            legal_actions = game_state.get_legal_actions()
            action = random.choice(legal_actions)
            game_state.apply_action(action)
            move_count += 1
            continue
        
        # Temperature schedule
        if move_count < temp_threshold:
            temperature = BOOTSTRAP_TEMP_INIT if is_bootstrap else TEMP_INIT
        else:
            temperature = BOOTSTRAP_TEMP_FINAL if is_bootstrap else TEMP_FINAL
        
        action_probs, _ = mcts.get_action_probs(game_state, temperature=temperature)
        
        # Store training data
        encoded_state = encode_state(game_state)
        player = game_state.current_player
        game_history.append((encoded_state, action_probs, player))
        
        # Select action
        if temperature == 0:
            action_idx = np.argmax(action_probs)
        else:
            action_idx = np.random.choice(len(action_probs), p=action_probs)
        
        action = index_to_action(action_idx)
        game_state.apply_action(action)
        move_count += 1
    
    # Assign values
    winner = game_state.winner
    
    examples = []
    for state, policy, player in game_history:
        if winner is None:
            value = -0.2 if not is_bootstrap else 0.0
        elif winner == player:
            value = 1.0
        else:
            value = -1.0
        
        # Convert to dict for serialization
        examples.append({
            'state': state,
            'policy': policy,
            'value': value
        })
    
    return examples, move_count


def augment_example_dict(example: dict) -> list[dict]:
    """Augment a training example dict with horizontal flip."""
    examples = [example]
    
    # Horizontal flip
    flipped_state = np.flip(example['state'], axis=2).copy()
    flipped_policy = _flip_policy(example['policy'])
    
    examples.append({
        'state': flipped_state,
        'policy': flipped_policy,
        'value': example['value']
    })
    
    return examples


def _flip_policy(policy: np.ndarray) -> np.ndarray:
    """Flip policy probabilities for horizontal board flip."""
    flipped = np.zeros_like(policy)
    
    # Flip move actions (indices 0-80, 9x9 grid)
    for idx in range(81):
        row = idx // 9
        col = idx % 9
        new_col = 8 - col
        new_idx = row * 9 + new_col
        flipped[new_idx] = policy[idx]
    
    # Flip horizontal walls (indices 81-144, 8x8 grid)
    for idx in range(81, 145):
        wall_idx = idx - 81
        row = wall_idx // 8
        col = wall_idx % 8
        new_col = 7 - col
        new_idx = 81 + row * 8 + new_col
        flipped[new_idx] = policy[idx]
    
    # Flip vertical walls (indices 145-208, 8x8 grid)
    for idx in range(145, 209):
        wall_idx = idx - 145
        row = wall_idx // 8
        col = wall_idx % 8
        new_col = 7 - col
        new_idx = 145 + row * 8 + new_col
        flipped[new_idx] = policy[idx]
    
    return flipped


# =============================================================================
# Distributed Trainer
# =============================================================================

class DistributedTrainer:
    """
    Orchestrates distributed training across CPU and GPU machines.
    
    Architecture:
    - Runs on CPU head node
    - Spawns InferenceServer and TrainingActor on GPU node
    - Spawns self-play tasks on CPU node
    - Coordinates data flow and weight synchronization
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "runs",
        num_workers: int = RAY_NUM_WORKERS,
        local_mode: bool = RAY_LOCAL_MODE
    ):
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.num_workers = num_workers
        self.local_mode = local_mode
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize Ray
        self._init_ray()
        
        # Training state
        self.current_iteration = 0
        self.best_win_rate = 0.0
        self.curriculum_stage = "bootstrap_random"
        self.shutdown_requested = False
        self.training_start_time = None
        self.last_save_time = None
        
        # Ray actors (initialized lazily)
        self.inference_server = None
        self.training_actor = None
        self.replay_buffer = None
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _init_ray(self):
        """Initialize Ray cluster connection."""
        if ray.is_initialized():
            print("[DistributedTrainer] Ray already initialized")
            return
        
        if self.local_mode:
            print("[DistributedTrainer] Starting Ray in LOCAL MODE")
            try:
                # For local mode, don't use working_dir - just run in current process
                ray.init(
                    num_cpus=self.num_workers + 2,
                    num_gpus=1 if torch.cuda.is_available() else 0,
                    include_dashboard=False,
                    ignore_reinit_error=True,
                )
            except Exception as e:
                print(f"[DistributedTrainer] Local init warning: {e}")
                ray.init(ignore_reinit_error=True)
        else:
            print(f"[DistributedTrainer] Connecting to Ray cluster at {RAY_HEAD_ADDRESS}")
            try:
                # For cluster mode, workers should have the code installed via pip/uv
                # Don't auto-upload working directory
                ray.init(
                    address=RAY_HEAD_ADDRESS,
                    ignore_reinit_error=True,
                )
                print(f"[DistributedTrainer] Connected to cluster")
                print(f"  Available resources: {ray.available_resources()}")
            except Exception as e:
                print(f"[DistributedTrainer] Warning: Could not connect to cluster: {e}")
                print("[DistributedTrainer] Starting Ray locally")
                ray.init(ignore_reinit_error=True)
    
    def _setup_signal_handlers(self):
        """Setup handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            if not self.shutdown_requested:
                print(f"\n\nReceived signal {signum}. Finishing current iteration...")
                self.shutdown_requested = True
            else:
                print("\nForce quitting...")
                sys.exit(1)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _init_actors(self, resume_weights: Optional[dict] = None):
        """Initialize Ray actors."""
        print("\n[DistributedTrainer] Initializing actors...")
        
        # Create inference server on GPU
        self.inference_server = InferenceServer.remote(resume_weights)
        print("  ✓ InferenceServer created")
        
        # Create training actor on GPU
        self.training_actor = TrainingActor.remote(
            resume_weights,
            self.checkpoint_dir,
            self.log_dir
        )
        print("  ✓ TrainingActor created")
        
        # Create replay buffer (can run anywhere)
        self.replay_buffer = ReplayBufferActor.remote(REPLAY_BUFFER_SIZE)
        print("  ✓ ReplayBufferActor created")
        
        # Wait for actors to be ready
        ray.get(self.inference_server.get_stats.remote())
        ray.get(self.training_actor.get_state.remote())
        print("  ✓ All actors ready\n")
    
    def _run_self_play(
        self, 
        num_games: int, 
        num_simulations: int,
        is_bootstrap: bool = False
    ) -> tuple[int, float]:
        """
        Run parallel self-play games.
        
        Returns:
            Tuple of (total examples added, average game length)
        """
        # Launch all games in parallel
        futures = [
            self_play_game_distributed.remote(
                self.inference_server,
                num_simulations=num_simulations,
                temp_threshold=BOOTSTRAP_TEMP_THRESHOLD if is_bootstrap else TEMP_THRESHOLD,
                is_bootstrap=is_bootstrap
            )
            for _ in range(num_games)
        ]
        
        # Collect results as they complete
        total_examples = 0
        total_moves = 0
        games_completed = 0
        
        while futures:
            done, futures = ray.wait(futures, num_returns=1)
            
            for future in done:
                try:
                    examples_dicts, moves = ray.get(future)
                    
                    # Augment examples
                    augmented = []
                    for ex_dict in examples_dicts:
                        augmented.extend(augment_example_dict(ex_dict))
                    
                    # Convert to TrainingExample objects
                    examples = [
                        TrainingExample(
                            state=d['state'],
                            policy=d['policy'],
                            value=d['value']
                        )
                        for d in augmented
                    ]
                    
                    # Add to replay buffer
                    ray.get(self.replay_buffer.add_examples.remote(examples))
                    
                    total_examples += len(examples)
                    total_moves += moves
                    games_completed += 1
                    
                    print(f"\r  Games: {games_completed}/{num_games}, "
                          f"Examples: {total_examples}", end="", flush=True)
                    
                except Exception as e:
                    print(f"\n  Warning: Game failed: {e}")
        
        print()  # Newline after progress
        
        avg_length = total_moves / games_completed if games_completed > 0 else 0
        return total_examples, avg_length
    
    def _run_training(self, sample_size: int, epochs: int = 10) -> dict:
        """Run training on sampled examples."""
        # Sample from replay buffer
        examples = ray.get(self.replay_buffer.sample.remote(sample_size))
        
        if not examples:
            return {}
        
        # Train
        metrics = ray.get(self.training_actor.train_on_examples.remote(examples, epochs))
        
        # Update learning rate
        current_lr = ray.get(self.training_actor.step_scheduler.remote())
        metrics['learning_rate'] = current_lr
        
        return metrics
    
    def _sync_weights_to_inference(self):
        """Sync weights from training actor to inference server."""
        weights = ray.get(self.training_actor.get_weights.remote())
        ray.get(self.inference_server.update_weights.remote(weights))
    
    def _evaluate_against_random(self, num_games: int = EVAL_GAMES) -> float:
        """Evaluate network against random player."""
        # Run evaluation games
        futures = []
        for game_idx in range(num_games):
            futures.append(
                self_play_game_distributed.remote(
                    self.inference_server,
                    num_simulations=100,  # Reduced for speed
                    temp_threshold=0,  # Greedy
                    is_bootstrap=True  # Network vs random
                )
            )
        
        wins = 0
        total = 0
        
        for future in futures:
            try:
                examples, moves = ray.get(future)
                # Check if network won (value of last example from network's perspective)
                if examples:
                    last_value = examples[-1]['value']
                    if last_value > 0:
                        wins += 1
                total += 1
            except:
                pass
        
        return wins / total if total > 0 else 0.0
    
    def train(
        self,
        num_iterations: int = TRAINING_ITERATIONS,
        games_per_iteration: int = SELF_PLAY_GAMES,
        simulations_per_move: int = ALPHA_MCTS_SIMULATIONS,
        resume_from: Optional[str] = None
    ):
        """
        Main distributed training loop.
        """
        # Load checkpoint if resuming
        resume_weights = None
        if resume_from and os.path.exists(resume_from):
            print(f"[DistributedTrainer] Loading checkpoint from {resume_from}")
            checkpoint = torch.load(resume_from, map_location='cpu')
            resume_weights = checkpoint.get('state_dict')
            self.current_iteration = checkpoint.get('iteration', 0)
            self.best_win_rate = checkpoint.get('best_win_rate', 0.0)
            self.curriculum_stage = checkpoint.get('curriculum_stage', 'bootstrap_random')
        
        # Initialize actors
        self._init_actors(resume_weights)
        
        # Load replay buffer if exists
        buffer_path = os.path.join(self.checkpoint_dir, "replay_buffer.pkl")
        if os.path.exists(buffer_path):
            loaded = ray.get(self.replay_buffer.load.remote(buffer_path))
            print(f"[DistributedTrainer] Loaded {loaded} examples from replay buffer")
        
        self.training_start_time = time.time()
        self.last_save_time = time.time()
        start_iteration = self.current_iteration + 1
        
        print(f"\n{'='*60}")
        print(f"Starting Distributed AlphaZero Training")
        print(f"{'='*60}")
        print(f"Local Mode: {self.local_mode}")
        print(f"Curriculum Stage: {self.curriculum_stage.upper()}")
        print(f"Iterations: {start_iteration} to {num_iterations}")
        print(f"Games/iteration: {games_per_iteration}")
        print(f"MCTS simulations: {simulations_per_move}")
        print(f"Workers: {self.num_workers}")
        print(f"{'='*60}\n")
        
        try:
            for iteration in range(start_iteration, num_iterations + 1):
                if self.shutdown_requested:
                    break
                
                self.current_iteration = iteration
                ray.get(self.training_actor.set_iteration.remote(iteration))
                iter_start = time.time()
                
                print(f"\n{'='*50}")
                print(f"Iteration {iteration}/{num_iterations} - Stage: {self.curriculum_stage.upper()}")
                print(f"{'='*50}")
                
                # --- Self-Play Phase ---
                is_bootstrap = self.curriculum_stage == "bootstrap_random"
                stage_name = "Bootstrap" if is_bootstrap else "Self-Play"
                print(f"\n--- {stage_name} Phase ---")
                
                sp_start = time.time()
                num_examples, avg_length = self._run_self_play(
                    num_games=games_per_iteration,
                    num_simulations=simulations_per_move,
                    is_bootstrap=is_bootstrap
                )
                sp_time = time.time() - sp_start
                
                buffer_size = ray.get(self.replay_buffer.size.remote())
                print(f"  Time: {sp_time:.1f}s, Examples: {num_examples}, "
                      f"Avg length: {avg_length:.1f}")
                print(f"  Replay buffer: {buffer_size}")
                
                # Log to TensorBoard
                ray.get(self.training_actor.log_scalar.remote(
                    'SelfPlay/GameLength', avg_length, iteration
                ))
                ray.get(self.training_actor.log_scalar.remote(
                    'SelfPlay/BufferSize', buffer_size, iteration
                ))
                
                # --- Training Phase ---
                print(f"\n--- Training Phase ---")
                train_start = time.time()
                
                sample_size = min(buffer_size, BATCH_SIZE * 100)
                metrics = self._run_training(sample_size, epochs=10)
                
                train_time = time.time() - train_start
                
                if metrics:
                    print(f"  Time: {train_time:.1f}s, LR: {metrics.get('learning_rate', 0):.6f}")
                    print(f"  Policy Loss: {np.mean(metrics.get('policy_loss', [0])):.4f}")
                    print(f"  Value Loss: {np.mean(metrics.get('value_loss', [0])):.4f}")
                    
                    ray.get(self.training_actor.log_metrics.remote(metrics, iteration))
                
                # Sync weights to inference server
                self._sync_weights_to_inference()
                
                # --- Evaluation Phase ---
                if iteration % EVAL_INTERVAL == 0:
                    print(f"\n--- Evaluation Phase ---")
                    eval_start = time.time()
                    
                    win_rate = self._evaluate_against_random(EVAL_GAMES)
                    
                    eval_time = time.time() - eval_start
                    print(f"  Win rate vs random: {win_rate*100:.1f}% ({eval_time:.1f}s)")
                    
                    ray.get(self.training_actor.log_scalar.remote(
                        'Eval/WinRateVsRandom', win_rate, iteration
                    ))
                    
                    # Check curriculum switch
                    if self.curriculum_stage == "bootstrap_random" and win_rate >= WIN_RATE_THRESHOLD:
                        print(f"\n{'='*50}")
                        print(f"🎉 CURRICULUM STAGE SWITCH 🎉")
                        print(f"Win rate {win_rate*100:.1f}% >= {WIN_RATE_THRESHOLD*100:.0f}%")
                        print(f"Switching from BOOTSTRAP to SELF-PLAY")
                        print(f"{'='*50}\n")
                        
                        self.curriculum_stage = "self_play"
                        
                        # Save switch checkpoint
                        self._save_checkpoint(f"model_switch_iter_{iteration}.pt")
                    
                    # Save best model
                    if win_rate > self.best_win_rate:
                        self.best_win_rate = win_rate
                        ray.get(self.training_actor.set_best_win_rate.remote(win_rate))
                        self._save_checkpoint(f"model_iter_{iteration}.pt", is_best=True)
                
                # --- Checkpointing ---
                if iteration % CHECKPOINT_INTERVAL == 0:
                    self._save_checkpoint(f"model_iter_{iteration}.pt")
                    self.last_save_time = time.time()
                
                # Auto-save
                elif time.time() - self.last_save_time >= AUTO_SAVE_MINUTES * 60:
                    self._save_checkpoint("model_autosave.pt")
                    self.last_save_time = time.time()
                
                iter_time = time.time() - iter_start
                print(f"\nIteration {iteration} completed in {iter_time:.1f}s")
        
        except Exception as e:
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            self._cleanup()
    
    def _save_checkpoint(self, filename: str, is_best: bool = False):
        """Save training checkpoint."""
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        extra_state = {
            'curriculum_stage': self.curriculum_stage,
            'win_rate_threshold': WIN_RATE_THRESHOLD,
        }
        
        ray.get(self.training_actor.save_checkpoint.remote(filepath, extra_state))
        
        # Save replay buffer
        buffer_path = os.path.join(self.checkpoint_dir, "replay_buffer.pkl")
        ray.get(self.replay_buffer.save.remote(buffer_path))
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "model_best.pt")
            # Copy the weights
            weights = ray.get(self.training_actor.get_weights.remote())
            state = ray.get(self.training_actor.get_state.remote())
            torch.save({
                'state_dict': weights,
                'iteration': state['iteration'],
                'best_win_rate': state['best_win_rate'],
                'curriculum_stage': self.curriculum_stage,
            }, best_path)
            print(f"[DistributedTrainer] New best model saved")
    
    def _cleanup(self):
        """Cleanup resources."""
        print("\n" + "="*50)
        print("Saving final checkpoint...")
        
        try:
            self._save_checkpoint("model_final.pt")
        except Exception as e:
            print(f"Warning: Could not save final checkpoint: {e}")
        
        # Cleanup actors
        if self.training_actor:
            try:
                ray.get(self.training_actor.cleanup.remote())
            except:
                pass
        
        if self.shutdown_requested:
            print("Training interrupted by user.")
        else:
            print("Training Complete!")
        
        if self.training_start_time:
            total_time = time.time() - self.training_start_time
            print(f"Total training time: {total_time/3600:.2f} hours")
        
        print("="*50)


# =============================================================================
# Entry Point
# =============================================================================

def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint to resume from."""
    import re
    
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    
    # Priority: autosave > latest iteration > model_final
    autosave_path = os.path.join(checkpoint_dir, "model_autosave.pt")
    if os.path.exists(autosave_path):
        return autosave_path
    
    # Look for iteration checkpoints
    iter_checkpoints = [f for f in checkpoints if f.startswith('model_iter_')]
    if iter_checkpoints:
        def get_iter_num(filename):
            match = re.search(r'model_iter_(\d+)\.pt', filename)
            return int(match.group(1)) if match else 0
        
        iter_checkpoints.sort(key=get_iter_num, reverse=True)
        return os.path.join(checkpoint_dir, iter_checkpoints[0])
    
    return None


if __name__ == "__main__":
    print("="*60)
    print("AlphaZero Distributed Training for Quoridor")
    print("="*60)
    print("\nPress Ctrl+C to safely stop and save checkpoint.\n")
    
    checkpoint_dir = "checkpoints"
    resume_path = find_latest_checkpoint(checkpoint_dir)
    
    if resume_path:
        print(f"Found checkpoint: {resume_path}")
    
    trainer = DistributedTrainer(
        checkpoint_dir=checkpoint_dir,
        log_dir="runs",
        num_workers=RAY_NUM_WORKERS,
        local_mode=RAY_LOCAL_MODE
    )
    
    trainer.train(
        num_iterations=TRAINING_ITERATIONS,
        games_per_iteration=SELF_PLAY_GAMES,
        simulations_per_move=ALPHA_MCTS_SIMULATIONS,
        resume_from=resume_path
    )
