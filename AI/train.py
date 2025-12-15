"""
AlphaZero Training Pipeline for Quoridor
Production-ready training with:
- Parallelized self-play
- Mixed precision training
- Graceful shutdown handling
- TensorBoard logging
- Automatic checkpointing
- Data augmentation
"""
from __future__ import annotations
import os
import sys
import signal
import pickle
import atexit
import re

# Add parent directory to path for imports when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import time
import queue
from collections import deque
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# Use newer torch.amp API (works with PyTorch 2.0+)
try:
    from torch.amp import autocast, GradScaler
    AMP_DEVICE_TYPE = 'cuda'  # For newer API
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_DEVICE_TYPE = None  # For older API
import torch.multiprocessing as mp

# TensorBoard (optional)
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("TensorBoard not available. Install with: pip install tensorboard")

from game import GameState
from AI.network import QuoridorNet, create_network
from AI.alpha_mcts import AlphaMCTS
from AI.encoder import encode_state
from AI.action_utils import action_to_index, get_legal_action_mask, index_to_action

from constants import (
    SELF_PLAY_GAMES, REPLAY_BUFFER_SIZE, BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY, TRAINING_ITERATIONS,
    CHECKPOINT_INTERVAL, TEMP_THRESHOLD, TEMP_INIT, TEMP_FINAL,
    ALPHA_MCTS_SIMULATIONS, ACTION_SPACE_SIZE, GRID_COUNT,
    NUM_PARALLEL_GAMES, AUTO_SAVE_MINUTES, EVAL_INTERVAL, EVAL_GAMES,
    WIN_RATE_THRESHOLD, BOOTSTRAP_TEMP_INIT, BOOTSTRAP_TEMP_FINAL, BOOTSTRAP_TEMP_THRESHOLD
)


# --- TRAINING EXAMPLE ---

@dataclass
class TrainingExample:
    """A single training example from self-play."""
    state: np.ndarray          # Encoded board state
    policy: np.ndarray         # MCTS visit count distribution
    value: float               # Game outcome from this player's perspective


# --- REPLAY BUFFER ---

class ReplayBuffer:
    """
    Fixed-size buffer to store training examples from self-play.
    Supports saving/loading for training resumption.
    """
    
    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add(self, example: TrainingExample) -> None:
        """Add a training example to the buffer."""
        self.buffer.append(example)
    
    def add_batch(self, examples: list[TrainingExample]) -> None:
        """Add multiple training examples."""
        for ex in examples:
            self.buffer.append(ex)
    
    def sample(self, batch_size: int) -> list[TrainingExample]:
        """Sample a random batch of examples."""
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self) -> None:
        """Clear all examples."""
        self.buffer.clear()
    
    def save(self, filepath: str) -> None:
        """Save replay buffer to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.buffer), f)
    
    def load(self, filepath: str) -> None:
        """Load replay buffer from file."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                examples = pickle.load(f)
                self.buffer = deque(examples, maxlen=self.capacity)


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


# --- DATA AUGMENTATION ---

def augment_example(example: TrainingExample) -> list[TrainingExample]:
    """
    Augment training example with horizontal flip.
    Quoridor has left-right symmetry.
    
    Returns:
        List containing original and flipped examples
    """
    examples = [example]
    
    # Horizontal flip
    flipped_state = np.flip(example.state, axis=2).copy()
    flipped_policy = flip_policy(example.policy)
    
    examples.append(TrainingExample(
        state=flipped_state,
        policy=flipped_policy,
        value=example.value
    ))
    
    return examples


def flip_policy(policy: np.ndarray) -> np.ndarray:
    """
    Flip policy probabilities for horizontal board flip.
    
    Action space:
    - Indices 0-80: Move actions (9x9 grid positions)
    - Indices 81-144: Horizontal walls (8x8 positions)
    - Indices 145-208: Vertical walls (8x8 positions)
    """
    flipped = np.zeros_like(policy)
    
    # Flip move actions (indices 0-80, 9x9 grid)
    for idx in range(81):
        row = idx // 9
        col = idx % 9
        new_col = 8 - col  # Flip column
        new_idx = row * 9 + new_col
        flipped[new_idx] = policy[idx]
    
    # Flip horizontal walls (indices 81-144, 8x8 grid)
    for idx in range(81, 145):
        wall_idx = idx - 81
        row = wall_idx // 8
        col = wall_idx % 8
        new_col = 7 - col  # Flip column (walls span 2 cells)
        new_idx = 81 + row * 8 + new_col
        flipped[new_idx] = policy[idx]
    
    # Flip vertical walls (indices 145-208, 8x8 grid)
    for idx in range(145, 209):
        wall_idx = idx - 145
        row = wall_idx // 8
        col = wall_idx % 8
        new_col = 7 - col  # Flip column
        new_idx = 145 + row * 8 + new_col
        flipped[new_idx] = policy[idx]
    
    return flipped


def augment_examples(examples: list[TrainingExample]) -> list[TrainingExample]:
    """Augment a batch of examples with horizontal flips."""
    augmented = []
    for ex in examples:
        augmented.extend(augment_example(ex))
    return augmented


# --- SELF-PLAY ---

def self_play_game(
    network: QuoridorNet,
    num_simulations: int = ALPHA_MCTS_SIMULATIONS,
    temp_threshold: int = TEMP_THRESHOLD,
    verbose: bool = False
) -> tuple[list[TrainingExample], int]:
    """
    Play one game of self-play and collect training data.
    
    Args:
        network: Neural network for MCTS
        num_simulations: MCTS simulations per move
        temp_threshold: Move number after which temperature drops
        verbose: Print game progress
    
    Returns:
        Tuple of (training examples, number of moves)
    """
    # Ensure network is in eval mode for inference
    network.eval()
    
    mcts = AlphaMCTS(
        network=network,
        num_simulations=num_simulations,
        add_noise=True  # Exploration noise during self-play
    )
    
    game_state = GameState()
    game_history = []  # (state, policy, current_player)
    move_count = 0
    max_moves = 200  # Increased to allow proper game completion (Quoridor typically ends in 30-50 moves)
    
    while not game_state.is_terminal() and move_count < max_moves:
        # Temperature schedule: high early for exploration, low later
        if move_count < temp_threshold:
            temperature = TEMP_INIT
        else:
            temperature = TEMP_FINAL
        
        # Get MCTS policy
        action_probs, _ = mcts.get_action_probs(game_state, temperature=temperature)
        
        # Store training data (before move)
        # Note: We use encode_state (not canonical) to match MCTS policy coordinates
        encoded_state = encode_state(game_state)
        current_player = game_state.current_player
        game_history.append((encoded_state, action_probs, current_player))
        
        # Select and apply action
        if temperature == 0:
            action_idx = np.argmax(action_probs)
        else:
            action_idx = np.random.choice(len(action_probs), p=action_probs)
        
        action = index_to_action(action_idx)
        game_state.apply_action(action)
        
        move_count += 1
        
        if verbose and move_count % 10 == 0:
            print(f"Move {move_count}, Player {game_state.current_player}'s turn")
    
    # Game finished - assign values based on outcome
    winner = game_state.winner
    
    training_examples = []
    for state, policy, player in game_history:
        # Value from this player's perspective
        # IMPORTANT: Values must be in [-1, 1] to match tanh output of value head
        if winner is None:
            value = -0.2  # Draw
        elif winner == player:
            value = 1.0  # Win
        else:
            value = -1.0  # Loss
        
        training_examples.append(TrainingExample(
            state=state,
            policy=policy,
            value=value
        ))
    
    if verbose:
        print(f"Game finished in {move_count} moves. Winner: Player {winner}")
    
    return training_examples, move_count


def self_play_worker(
    worker_id: int,
    network_state_dict: dict,
    num_games: int,
    num_simulations: int,
    result_queue: mp.Queue,
    shutdown_event: mp.Event,
    device: str = 'cpu'
):
    """
    Worker function for parallel self-play.
    
    NOTE: When using GPU, multiprocessing has significant overhead.
    Consider using sequential self-play with GPU for better performance.
    
    Args:
        worker_id: Worker identifier
        network_state_dict: Network weights to load
        num_games: Number of games to play
        num_simulations: MCTS simulations per move
        result_queue: Queue to send results
        shutdown_event: Event to signal shutdown
        device: Device to use ('cpu' or 'cuda')
    """
    try:
        # Create network and load weights
        network = QuoridorNet()
        network.load_state_dict(network_state_dict)
        network.to(device)
        network.eval()
        
        all_examples = []
        total_moves = 0
        games_played = 0
        
        for game_idx in range(num_games):
            # Check for shutdown signal
            if shutdown_event.is_set():
                break
                
            examples, moves = self_play_game(
                network=network,
                num_simulations=num_simulations,
                verbose=False
            )
            all_examples.extend(examples)
            total_moves += moves
            games_played += 1
        
        # Send results back
        result_queue.put((worker_id, all_examples, total_moves, games_played))
    except KeyboardInterrupt:
        # Worker was interrupted - send partial results
        result_queue.put((worker_id, [], 0, 0))


def generate_self_play_data_parallel(
    network: QuoridorNet,
    num_games: int = SELF_PLAY_GAMES,
    num_simulations: int = ALPHA_MCTS_SIMULATIONS,
    num_workers: int = NUM_PARALLEL_GAMES,
    verbose: bool = True
) -> tuple[list[TrainingExample], float]:
    """
    Generate training data through parallel self-play.
    
    Args:
        network: Neural network for MCTS
        num_games: Total number of self-play games
        num_simulations: MCTS simulations per move
        num_workers: Number of parallel workers
        verbose: Print progress
    
    Returns:
        Tuple of (all training examples, average game length)
    """
    # For small number of games or single worker, use sequential
    if num_workers <= 1 or num_games < num_workers:
        all_examples = []
        total_moves = 0
        for game_idx in range(num_games):
            if verbose:
                print(f"\rSelf-play game {game_idx + 1}/{num_games}", end="")
            examples, moves = self_play_game(
                network=network,
                num_simulations=num_simulations,
                verbose=False
            )
            all_examples.extend(examples)
            total_moves += moves
        if verbose:
            print()
        avg_length = total_moves / num_games if num_games > 0 else 0
        return all_examples, avg_length
    
    # Parallel execution
    network_state = network.state_dict()
    games_per_worker = num_games // num_workers
    extra_games = num_games % num_workers
    
    # Use spawn context for CUDA compatibility
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    shutdown_event = ctx.Event()  # Shared shutdown signal
    processes = []
    
    try:
        for i in range(num_workers):
            worker_games = games_per_worker + (1 if i < extra_games else 0)
            p = ctx.Process(
                target=self_play_worker,
                args=(i, network_state, worker_games, num_simulations, result_queue, shutdown_event, 'cpu')
            )
            processes.append(p)
            p.start()
        
        # Collect results
        all_examples = []
        total_moves = 0
        total_games = 0
        
        for _ in range(num_workers):
            try:
                worker_id, examples, moves, games = result_queue.get(timeout=5)
                all_examples.extend(examples)
                total_moves += moves
                total_games += games
                if verbose and games > 0:
                    print(f"  Worker {worker_id} completed {games} games")
            except queue.Empty:
                break
        
        # Wait for all processes with timeout
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
    
    except KeyboardInterrupt:
        # Signal workers to shutdown
        shutdown_event.set()
        for p in processes:
            if p.is_alive():
                p.terminate()
        raise
    
    finally:
        # Ensure all processes are cleaned up
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)
    
    avg_length = total_moves / total_games if total_games > 0 else 0
    return all_examples, avg_length


def generate_self_play_data(
    network: QuoridorNet,
    num_games: int = SELF_PLAY_GAMES,
    num_simulations: int = ALPHA_MCTS_SIMULATIONS,
    verbose: bool = True,
    check_interrupt: callable = None
) -> tuple[list[TrainingExample], float]:
    """
    Generate training data through self-play (sequential version).
    
    Args:
        check_interrupt: Optional function that returns True if should stop
    
    Returns:
        Tuple of (all training examples, average game length)
    """
    all_examples = []
    total_moves = 0
    games_played = 0
    
    for game_idx in range(num_games):
        # Check for interrupt signal
        if check_interrupt and check_interrupt():
            break
            
        if verbose:
            print(f"\rSelf-play game {game_idx + 1}/{num_games}", end="", flush=True)
        
        examples, moves = self_play_game(
            network=network,
            num_simulations=num_simulations,
            verbose=False
        )
        all_examples.extend(examples)
        total_moves += moves
        games_played += 1
    
    if verbose:
        print(f" - Generated {len(all_examples)} examples")
    
    avg_length = total_moves / games_played if games_played > 0 else 0
    return all_examples, avg_length


# --- BOOTSTRAP VS RANDOM ---

def network_vs_random_game(
    network: QuoridorNet,
    num_simulations: int = ALPHA_MCTS_SIMULATIONS,
    temp_threshold: int = BOOTSTRAP_TEMP_THRESHOLD,
    verbose: bool = False
) -> tuple[list[TrainingExample], int]:
    """
    Play one game where network plays vs random opponent.
    Collect training examples only on network's turns.
    Uses temperature schedule: high early for exploration, low later for exploitation.
    
    Args:
        network: Neural network for MCTS
        num_simulations: MCTS simulations per network move
        temp_threshold: Move number after which temperature drops
        verbose: Print game progress
    
    Returns:
        Tuple of (training examples, number of moves)
    """
    network.eval()
    
    mcts = AlphaMCTS(
        network=network,
        num_simulations=num_simulations,
        add_noise=True  # Add exploration noise during bootstrap
    )
    
    game_state = GameState()
    game_history = []  # (state, policy, network_player)
    move_count = 0
    max_moves = 200  # Increased to allow proper game completion
    
    # Randomly decide which player is the network
    network_player = 2
    
    while not game_state.is_terminal() and move_count < max_moves:
        current = game_state.current_player
        
        if current == network_player:
            # Network's turn - use MCTS with temperature schedule
            # High temperature early for exploration, low later for exploitation
            if move_count < temp_threshold:
                temperature = BOOTSTRAP_TEMP_INIT
            else:
                temperature = BOOTSTRAP_TEMP_FINAL
            
            action_probs, _ = mcts.get_action_probs(game_state, temperature=temperature)
            
            # Store training example
            # Note: We use encode_state (not canonical) to match MCTS policy coordinates
            encoded_state = encode_state(game_state)
            game_history.append((encoded_state, action_probs, network_player))
            
            # Select action
            if temperature == 0:
                action_idx = np.argmax(action_probs)
            else:
                action_idx = np.random.choice(len(action_probs), p=action_probs)
            
            action = index_to_action(action_idx)
        else:
            # Random opponent's turn
            legal_actions = game_state.get_legal_actions()
            action = random.choice(legal_actions)
        
        game_state.apply_action(action)
        move_count += 1
        
        if verbose and move_count % 10 == 0:
            print(f"Move {move_count}, Player {game_state.current_player}'s turn")
    
    # Assign values based on outcome
    winner = game_state.winner
    
    training_examples = []
    for state, policy, player in game_history:
        if winner is None:
            value = 0.0
        elif winner == player:
            value = 1.0  # Network won
        else:
            value = -1.0  # Network lost
        
        training_examples.append(TrainingExample(
            state=state,
            policy=policy,
            value=value
        ))
    
    if verbose:
        result = "Won" if winner == network_player else ("Lost" if winner else "Draw")
        print(f"Game finished in {move_count} moves. Network {result}")
    
    return training_examples, move_count


def generate_network_vs_random_data(
    network: QuoridorNet,
    num_games: int = SELF_PLAY_GAMES,
    num_simulations: int = ALPHA_MCTS_SIMULATIONS,
    temp_threshold: int = BOOTSTRAP_TEMP_THRESHOLD,
    verbose: bool = True,
    check_interrupt: callable = None
) -> tuple[list[TrainingExample], float]:
    """
    Generate training data by playing network vs random opponent.
    
    Args:
        network: Neural network for MCTS
        num_games: Number of games to play
        num_simulations: MCTS simulations per network move
        temp_threshold: Move number to drop temperature
        check_interrupt: Optional function that returns True if should stop
        verbose: Print progress
    
    Returns:
        Tuple of (all training examples, average game length)
    """
    all_examples = []
    total_moves = 0
    games_played = 0
    
    for game_idx in range(num_games):
        # Check for interrupt signal
        if check_interrupt and check_interrupt():
            break
            
        if verbose:
            print(f"\rBootstrap game {game_idx + 1}/{num_games}", end="", flush=True)
        
        examples, moves = network_vs_random_game(
            network=network,
            num_simulations=num_simulations,
            temp_threshold=temp_threshold,
            verbose=False
        )
        all_examples.extend(examples)
        total_moves += moves
        games_played += 1
    
    if verbose:
        print(f" - Generated {len(all_examples)} examples")
    
    avg_length = total_moves / games_played if games_played > 0 else 0
    return all_examples, avg_length


# --- TRAINING ---

class Trainer:
    """
    Main training class with:
    - Mixed precision training
    - Learning rate scheduling
    - Graceful shutdown
    - TensorBoard logging
    - Automatic checkpointing
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "runs",
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_amp: bool = True,
        use_parallel: bool = True
    ):
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.device = device
        self.use_amp = use_amp and device == 'cuda'
        self.use_parallel = use_parallel
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize network
        self.network = create_network(device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # Learning rate scheduler (simple cosine annealing - no warm restarts during bootstrap)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=TRAINING_ITERATIONS,  # Decay over full training
            eta_min=LEARNING_RATE / 100
        )
        
        # Mixed precision scaler
        if self.use_amp:
            if AMP_DEVICE_TYPE:
                self.scaler = GradScaler(AMP_DEVICE_TYPE)
            else:
                self.scaler = GradScaler()
        else:
            self.scaler = None

        # Replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        
        # Training state
        self.current_iteration = 0
        self.best_win_rate = 0.0
        self.training_start_time = None
        self.last_save_time = None
        
        # Curriculum learning state
        self.curriculum_stage = "bootstrap_random"  # Start with bootstrap, switch to "self_play"
        self.win_rate_threshold = WIN_RATE_THRESHOLD  # Threshold to switch stages
        
        # Shutdown flag
        self.shutdown_requested = False
        
        # TensorBoard writer
        self.writer = None
        if HAS_TENSORBOARD:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.writer = SummaryWriter(os.path.join(log_dir, f"quoridor_{timestamp}"))
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            if not self.shutdown_requested:
                print(f"\n\nReceived signal {signum}. Finishing current iteration and saving...")
                print("Press Ctrl+C again to force quit (may lose progress)\n")
                self.shutdown_requested = True
            else:
                print("\nForce quitting...")
                sys.exit(1)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Register cleanup on exit
        atexit.register(self._cleanup)
    
    def _cleanup(self):
        """Cleanup resources on exit."""
        if self.writer:
            self.writer.close()
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save full training state."""
        checkpoint = {
            'iteration': self.current_iteration,
            'state_dict': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_win_rate': self.best_win_rate,
            'curriculum_stage': self.curriculum_stage,
            'win_rate_threshold': self.win_rate_threshold,
            'hyperparameters': {
                'learning_rate': LEARNING_RATE,
                'batch_size': BATCH_SIZE,
                'num_res_blocks': self.network.res_blocks.__len__() if hasattr(self.network, 'res_blocks') else 0,
                'num_filters': 128,
                'mcts_simulations': ALPHA_MCTS_SIMULATIONS,
            },
            'timestamp': datetime.now().isoformat(),
        }
        
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint to {filepath}")
        
        # Also save replay buffer
        buffer_path = filepath.replace('.pt', '_buffer.pkl')
        self.replay_buffer.save(buffer_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "model_best.pt")
            torch.save(checkpoint, best_path)
            print(f"New best model saved to {best_path}")
    
    def load_checkpoint(self, filepath: str) -> bool:
        """Load training state from checkpoint."""
        if not os.path.exists(filepath):
            return False
        
        print(f"Loading checkpoint from {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Check if checkpoint architecture matches current network
        try:
            self.network.load_state_dict(checkpoint['state_dict'])
        except RuntimeError as e:
            if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                print(f"Warning: Checkpoint architecture mismatch!")
                print(f"  Current network has {len(list(self.network.parameters()))} params")
                print(f"  This usually means NUM_RES_BLOCKS changed.")
                print(f"  Starting fresh training with new architecture.")
                return False
            raise
        
        # Only load optimizer state if architecture matched
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        except (ValueError, KeyError) as e:
            print(f"Warning: Could not load optimizer state: {e}")
            print("  Using fresh optimizer.")
        
        if 'scheduler_state' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not load scheduler state: {e}")
        
        self.current_iteration = checkpoint.get('iteration', 0)
        self.best_win_rate = checkpoint.get('best_win_rate', 0.0)
        
        # Load curriculum state (with backward compatibility)
        self.curriculum_stage = checkpoint.get('curriculum_stage', 'bootstrap_random')
        self.win_rate_threshold = checkpoint.get('win_rate_threshold', WIN_RATE_THRESHOLD)
        
        # Load replay buffer if exists
        buffer_path = filepath.replace('.pt', '_buffer.pkl')
        if os.path.exists(buffer_path):
            self.replay_buffer.load(buffer_path)
            print(f"Loaded {len(self.replay_buffer)} examples from replay buffer")
        
        print(f"Resumed from iteration {self.current_iteration}")
        print(f"Curriculum stage: {self.curriculum_stage}")
        return True
    
    def train_on_examples(
        self,
        examples: list[TrainingExample],
        epochs: int = 10,
        verbose: bool = True
    ) -> dict:
        """Train on collected examples with mixed precision."""
        self.network.train()
        
        # Create data loader
        dataset = TrainingDataset(examples)
        dataloader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
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
                
                # Mixed precision forward pass
                if self.use_amp:
                    if AMP_DEVICE_TYPE:
                        with autocast(AMP_DEVICE_TYPE):
                            policy_logits, pred_values = self.network(states)
                            
                            # Policy loss (cross-entropy with MCTS policy as target)
                            policy_loss = -torch.mean(
                                torch.sum(policies * torch.log_softmax(policy_logits, dim=1), dim=1)
                            )
                            
                            # Value loss (MSE)
                            value_loss = nn.functional.mse_loss(pred_values, values)
                            
                            # Total loss (increased value loss weight to 2.0)
                            total_loss = policy_loss + 2.0 * value_loss
                    else:
                        with autocast():
                            policy_logits, pred_values = self.network(states)
                            policy_loss = -torch.mean(
                                torch.sum(policies * torch.log_softmax(policy_logits, dim=1), dim=1)
                            )
                            value_loss = nn.functional.mse_loss(pred_values, values)
                            total_loss = policy_loss + 2.0 * value_loss
                    
                    # Scaled backward pass
                    self.scaler.scale(total_loss).backward()
                    # Gradient clipping for training stability
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard forward pass
                    policy_logits, pred_values = self.network(states)
                    
                    policy_loss = -torch.mean(
                        torch.sum(policies * torch.log_softmax(policy_logits, dim=1), dim=1)
                    )
                    value_loss = nn.functional.mse_loss(pred_values, values)
                    total_loss = policy_loss + 2.0 * value_loss
                    
                    total_loss.backward()
                    # Gradient clipping for training stability
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_total_loss += total_loss.item()
                num_batches += 1
            
            # Average losses for epoch
            if num_batches > 0:
                avg_policy = epoch_policy_loss / num_batches
                avg_value = epoch_value_loss / num_batches
                avg_total = epoch_total_loss / num_batches
                
                metrics['policy_loss'].append(avg_policy)
                metrics['value_loss'].append(avg_value)
                metrics['total_loss'].append(avg_total)
                
                if verbose and (epoch == 0 or epoch == epochs - 1):
                    print(f"  Epoch {epoch + 1}/{epochs}: "
                          f"Policy={avg_policy:.4f}, Value={avg_value:.4f}, Total={avg_total:.4f}")
        
        return metrics
    
    def evaluate_against_random(self, num_games: int = EVAL_GAMES) -> float:
        """Evaluate network against random player."""
        from AI.alpha_mcts import AlphaZeroPlayer
        
        self.network.eval()
        wins = 0
        
        for game_idx in range(num_games):
            network_is_p1 = (game_idx % 2 == 0)
            game_state = GameState()
            
            alpha_player = AlphaZeroPlayer(
                player=1 if network_is_p1 else 2,
                network=self.network,
                num_simulations=100,  # Reduced for faster evaluation
                temperature=0.0
            )
            
            while not game_state.is_terminal():
                current = game_state.current_player
                is_network_turn = (current == 1) == network_is_p1
                
                if is_network_turn:
                    action = alpha_player.get_action(game_state)
                else:
                    actions = game_state.get_legal_actions()
                    action = random.choice(actions)
                
                game_state.apply_action(action)
            
            winner = game_state.winner
            if winner is not None and ((winner == 1) == network_is_p1):
                wins += 1
        
        win_rate = wins / num_games
        return win_rate

    def evaluate_against_random_parallel(
        self,
        num_games: int = EVAL_GAMES,
        simulations: int = 100,
        num_workers: int = NUM_PARALLEL_GAMES,
        device: Optional[str] = None,
        verbose: bool = True
    ) -> dict:
        """Parallel evaluation vs random using multiprocessing."""
        return evaluate_against_random_parallel(
            network=self.network,
            num_games=num_games,
            simulations=simulations,
            num_workers=num_workers,
            device=device or self.device,
            verbose=verbose
        )
    
    def should_auto_save(self) -> bool:
        """Check if auto-save is needed based on time."""
        if self.last_save_time is None:
            return False
        elapsed = time.time() - self.last_save_time
        return elapsed >= AUTO_SAVE_MINUTES * 60
    
    def train(
        self,
        num_iterations: int = TRAINING_ITERATIONS,
        games_per_iteration: int = SELF_PLAY_GAMES,
        simulations_per_move: int = ALPHA_MCTS_SIMULATIONS,
        resume_from: Optional[str] = None
    ):
        """
        Main training loop.
        
        Args:
            num_iterations: Total training iterations
            games_per_iteration: Self-play games per iteration
            simulations_per_move: MCTS simulations per move
            resume_from: Path to checkpoint to resume from
        """
        # Resume from checkpoint if provided
        if resume_from:
            if not self.load_checkpoint(resume_from):
                print(f"Could not load checkpoint from {resume_from}, starting fresh")
        
        self.training_start_time = time.time()
        self.last_save_time = time.time()
        start_iteration = self.current_iteration + 1
        
        print(f"\n{'='*60}")
        print(f"Starting AlphaZero Training")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Parallel Self-Play: {self.use_parallel} ({NUM_PARALLEL_GAMES} workers)")
        print(f"Curriculum Stage: {self.curriculum_stage.upper()}")
        print(f"Win Rate Threshold: {self.win_rate_threshold*100:.0f}%")
        print(f"Iterations: {start_iteration} to {num_iterations}")
        print(f"Games/iteration: {games_per_iteration}")
        print(f"MCTS simulations: {simulations_per_move}")
        print(f"{'='*60}\n")
        
        try:
            for iteration in range(start_iteration, num_iterations + 1):
                if self.shutdown_requested:
                    break
                
                self.current_iteration = iteration
                iter_start = time.time()
                
                print(f"\n{'='*50}")
                print(f"Iteration {iteration}/{num_iterations} - Stage: {self.curriculum_stage.upper()}")
                print(f"{'='*50}")
                
                # --- Data Generation Phase ---
                print(f"\n--- {'Bootstrap' if self.curriculum_stage == 'bootstrap_random' else 'Self-Play'} Phase ---")
                sp_start = time.time()
                
                if self.curriculum_stage == "bootstrap_random":
                    # Bootstrap stage: Network vs Random
                    examples, avg_game_length = generate_network_vs_random_data(
                        network=self.network,
                        num_games=games_per_iteration,
                        num_simulations=simulations_per_move,
                        temp_threshold=BOOTSTRAP_TEMP_THRESHOLD,
                        verbose=True,
                        check_interrupt=lambda: self.shutdown_requested
                    )
                else:
                    # Self-play stage: Network vs Network
                    if self.use_parallel and NUM_PARALLEL_GAMES > 1:
                        examples, avg_game_length = generate_self_play_data_parallel(
                            network=self.network,
                            num_games=games_per_iteration,
                            num_simulations=simulations_per_move,
                            num_workers=NUM_PARALLEL_GAMES,
                            verbose=True
                        )
                    else:
                        examples, avg_game_length = generate_self_play_data(
                            network=self.network,
                            num_games=games_per_iteration,
                            num_simulations=simulations_per_move,
                            verbose=True,
                            check_interrupt=lambda: self.shutdown_requested
                        )
                
                # Augment with horizontal flips
                augmented_examples = augment_examples(examples)
                self.replay_buffer.add_batch(augmented_examples)
                
                sp_time = time.time() - sp_start
                stage_name = "Bootstrap" if self.curriculum_stage == "bootstrap_random" else "Self-play"
                print(f"{stage_name}: {sp_time:.1f}s, {len(examples)} examples "
                      f"(+{len(augmented_examples) - len(examples)} augmented)")
                print(f"Avg game length: {avg_game_length:.1f} moves")
                print(f"Replay buffer: {len(self.replay_buffer)} examples")
                
                # Log to TensorBoard
                if self.writer:
                    self.writer.add_scalar('SelfPlay/GameLength', avg_game_length, iteration)
                    self.writer.add_scalar('SelfPlay/BufferSize', len(self.replay_buffer), iteration)
                    self.writer.add_scalar('SelfPlay/TimeSeconds', sp_time, iteration)
                    self.writer.add_scalar('Curriculum/Stage', 0 if self.curriculum_stage == "bootstrap_random" else 1, iteration)
                
                # --- Training Phase ---
                print("\n--- Training Phase ---")
                train_start = time.time()
                
                # Sample from replay buffer
                sample_size = min(len(self.replay_buffer), BATCH_SIZE * 100)
                training_examples = self.replay_buffer.sample(sample_size)
                
                metrics = self.train_on_examples(
                    examples=training_examples,
                    epochs=10,
                    verbose=True
                )
                
                # Update learning rate
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                
                train_time = time.time() - train_start
                print(f"Training: {train_time:.1f}s, LR: {current_lr:.6f}")
                
                # Log training metrics
                if self.writer and metrics['total_loss']:
                    self.writer.add_scalar('Train/PolicyLoss', np.mean(metrics['policy_loss']), iteration)
                    self.writer.add_scalar('Train/ValueLoss', np.mean(metrics['value_loss']), iteration)
                    self.writer.add_scalar('Train/TotalLoss', np.mean(metrics['total_loss']), iteration)
                    self.writer.add_scalar('Train/LearningRate', current_lr, iteration)
                
                # --- Evaluation Phase ---
                # Evaluate every EVAL_INTERVAL iterations (regardless of curriculum stage)
                should_eval = (iteration % EVAL_INTERVAL == 0)
                
                if should_eval:
                    print("\n--- Evaluation Phase ---")
                    eval_start = time.time()
                    
                    if self.use_parallel and NUM_PARALLEL_GAMES > 1 and EVAL_GAMES >= NUM_PARALLEL_GAMES *5:
                        eval_results = self.evaluate_against_random_parallel(
                            num_games=EVAL_GAMES,
                            simulations=100,
                            num_workers=NUM_PARALLEL_GAMES,
                            device=self.device,
                            verbose=True
                        )
                        win_rate = eval_results['network_win_rate']
                    else:
                        win_rate = self.evaluate_against_random(num_games=EVAL_GAMES)
                        eval_results = None
                    
                    eval_time = time.time() - eval_start
                    print(f"Win rate vs random: {win_rate*100:.1f}% ({eval_time:.1f}s)")

                    if eval_results:
                        print(
                            f"  Games: {eval_results['games_played']}, "
                            f"Network: {eval_results['network_wins']}, "
                            f"Random: {eval_results['random_wins']}, "
                            f"Draws: {eval_results['draws']}"
                        )
                    
                    if self.writer:
                        self.writer.add_scalar('Eval/WinRateVsRandom', win_rate, iteration)
                    
                    # Check for curriculum stage switch
                    if self.curriculum_stage == "bootstrap_random" and win_rate >= self.win_rate_threshold:
                        print(f"\n{'='*50}")
                        print(f"🎉 CURRICULUM STAGE SWITCH 🎉")
                        print(f"Win rate {win_rate*100:.1f}% >= {self.win_rate_threshold*100:.0f}% threshold")
                        print(f"Switching from BOOTSTRAP to SELF-PLAY")
                        print(f"{'='*50}\n")
                        
                        self.curriculum_stage = "self_play"
                        
                        # Save checkpoint immediately after switching
                        switch_path = os.path.join(self.checkpoint_dir, f"model_switch_iter_{iteration}.pt")
                        self.save_checkpoint(switch_path, is_best=False)
                        print(f"Saved curriculum switch checkpoint")
                        
                        if self.writer:
                            self.writer.add_scalar('Curriculum/SwitchIteration', iteration, iteration)
                    
                    # Save best model
                    if win_rate > self.best_win_rate:
                        self.best_win_rate = win_rate
                        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_iter_{iteration}.pt")
                        self.save_checkpoint(checkpoint_path, is_best=True)
                
                # --- Checkpointing ---
                if iteration % CHECKPOINT_INTERVAL == 0:
                    checkpoint_path = os.path.join(self.checkpoint_dir, f"model_iter_{iteration}.pt")
                    self.save_checkpoint(checkpoint_path)
                    self.last_save_time = time.time()
                
                # Auto-save based on time
                elif self.should_auto_save():
                    checkpoint_path = os.path.join(self.checkpoint_dir, "model_autosave.pt")
                    self.save_checkpoint(checkpoint_path)
                    self.last_save_time = time.time()
                
                iter_time = time.time() - iter_start
                print(f"\nIteration {iteration} completed in {iter_time:.1f}s")
                
        except Exception as e:
            print(f"\nError during training: {e}")
            raise
        
        finally:
            # Always save on exit
            print("\n" + "="*50)
            print("Saving final checkpoint...")
            final_path = os.path.join(self.checkpoint_dir, "model_final.pt")
            self.save_checkpoint(final_path)
            
            if self.shutdown_requested:
                print("Training interrupted by user.")
            else:
                print("Training Complete!")
            
            total_time = time.time() - self.training_start_time
            print(f"Total training time: {total_time/3600:.2f} hours")
            print("="*50)


def evaluate_against_random(
    network: QuoridorNet,
    num_games: int = 20,
    simulations: int = 100
) -> dict:
    """
    Evaluate trained network against random player.
    
    Args:
        network: Trained neural network
        num_games: Number of evaluation games
        simulations: MCTS simulations per move
    
    Returns:
        Dictionary with win rate statistics
    """
    from AI.alpha_mcts import AlphaZeroPlayer
    
    wins = {'network': 0, 'random': 0, 'draws': 0}
    
    for game_idx in range(num_games):
        # Alternate who goes first
        network_is_p1 = (game_idx % 2 == 0)
        
        game_state = GameState()
        
        alpha_player = AlphaZeroPlayer(
            player=1 if network_is_p1 else 2,
            network=network,
            num_simulations=simulations,
            temperature=0.0  # Greedy play
        )
        
        while not game_state.is_terminal():
            current = game_state.current_player
            is_network_turn = (current == 1) == network_is_p1
            
            if is_network_turn:
                action = alpha_player.get_action(game_state)
            else:
                # Random player
                actions = game_state.get_legal_actions()
                action = random.choice(actions)
            
            game_state.apply_action(action)
        
        winner = game_state.winner
        if winner is None:
            wins['draws'] += 1
        elif (winner == 1) == network_is_p1:
            wins['network'] += 1
        else:
            wins['random'] += 1
    
    total = num_games
    results = {
        'network_wins': wins['network'],
        'random_wins': wins['random'],
        'draws': wins['draws'],
        'network_win_rate': wins['network'] / total,
        'games_played': total
    }
    
    print(f"\nEvaluation Results ({num_games} games):")
    print(f"  Network wins: {wins['network']} ({100*wins['network']/total:.1f}%)")
    print(f"  Random wins:  {wins['random']} ({100*wins['random']/total:.1f}%)")
    print(f"  Draws:        {wins['draws']}")
    
    return results


def _eval_worker(
    worker_id: int,
    network_state_dict: dict,
    num_games: int,
    simulations: int,
    result_queue: mp.Queue,
    device: str = 'cpu',
    seed_base: int | None = None
):
    """Worker process for parallel evaluation vs random."""
    try:
        seed = (seed_base or int(time.time())) + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        network = QuoridorNet()
        network.load_state_dict(network_state_dict)
        network.to(device)
        network.eval()

        from AI.alpha_mcts import AlphaZeroPlayer

        wins = {'network': 0, 'random': 0, 'draws': 0}

        for game_idx in range(num_games):
            network_is_p1 = (game_idx % 2 == 0)
            game_state = GameState()

            alpha_player = AlphaZeroPlayer(
                player=1 if network_is_p1 else 2,
                network=network,
                num_simulations=simulations,
                temperature=0.0
            )

            while not game_state.is_terminal():
                current = game_state.current_player
                is_network_turn = (current == 1) == network_is_p1

                if is_network_turn:
                    action = alpha_player.get_action(game_state)
                else:
                    actions = game_state.get_legal_actions()
                    action = random.choice(actions)

                game_state.apply_action(action)

            winner = game_state.winner
            if winner is None:
                wins['draws'] += 1
            elif (winner == 1) == network_is_p1:
                wins['network'] += 1
            else:
                wins['random'] += 1

        result_queue.put((worker_id, wins, num_games))
    except KeyboardInterrupt:
        result_queue.put((worker_id, {'network': 0, 'random': 0, 'draws': 0}, 0))


def evaluate_against_random_parallel(
    network: QuoridorNet,
    num_games: int = 20,
    simulations: int = 100,
    num_workers: int = NUM_PARALLEL_GAMES,
    device: str = 'cpu',
    verbose: bool = True
) -> dict:
    """Parallel evaluation vs random opponent."""
    if num_workers <= 1 or num_games < num_workers:
        return evaluate_against_random(network, num_games=num_games, simulations=simulations)

    network_state = network.state_dict()
    games_per_worker = num_games // num_workers
    extra_games = num_games % num_workers

    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    processes = []

    seed_base = int(time.time())

    for i in range(num_workers):
        worker_games = games_per_worker + (1 if i < extra_games else 0)
        if worker_games == 0:
            continue

        p = ctx.Process(
            target=_eval_worker,
            args=(i, network_state, worker_games, simulations, result_queue, device, seed_base)
        )
        p.start()
        processes.append(p)

    wins = {'network': 0, 'random': 0, 'draws': 0}
    total_games = 0

    for _ in processes:
        worker_id, worker_wins, worker_total = result_queue.get()
        wins['network'] += worker_wins.get('network', 0)
        wins['random'] += worker_wins.get('random', 0)
        wins['draws'] += worker_wins.get('draws', 0)
        total_games += worker_total

    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    total = total_games if total_games > 0 else num_games
    results = {
        'network_wins': wins['network'],
        'random_wins': wins['random'],
        'draws': wins['draws'],
        'network_win_rate': wins['network'] / total if total > 0 else 0.0,
        'games_played': total
    }

    if verbose:
        print(
            f"Parallel eval ({num_workers} workers) — "
            f"Network: {results['network_wins']}, "
            f"Random: {results['random_wins']}, "
            f"Draws: {results['draws']}"
        )

    return results


# Legacy function for backward compatibility
def training_loop(
    num_iterations: int = TRAINING_ITERATIONS,
    games_per_iteration: int = SELF_PLAY_GAMES,
    simulations_per_move: int = ALPHA_MCTS_SIMULATIONS,
    checkpoint_dir: str = "checkpoints",
    resume_from: Optional[str] = None,
    device: str = 'cpu'
) -> QuoridorNet:
    """
    Legacy training loop function for backward compatibility.
    Uses the new Trainer class internally.
    """
    trainer = Trainer(
        checkpoint_dir=checkpoint_dir,
        device=device,
        use_amp=(device == 'cuda'),
        use_parallel=True
    )
    
    trainer.train(
        num_iterations=num_iterations,
        games_per_iteration=games_per_iteration,
        simulations_per_move=simulations_per_move,
        resume_from=resume_from
    )
    
    return trainer.network


if __name__ == "__main__":
    # Production training
    print("="*60)
    print("AlphaZero Training for Quoridor - Production Mode")
    print("="*60)
    print("\nPress Ctrl+C to safely stop and save checkpoint.\n")
    
    # Use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Check for existing checkpoint to resume
    resume_path = None
    checkpoint_dir = "checkpoints"
    
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        
        # Priority: autosave > latest iteration > model_final
        autosave_path = os.path.join(checkpoint_dir, "model_autosave.pt")
        if os.path.exists(autosave_path):
            resume_path = autosave_path
            print(f"Found autosave checkpoint: {resume_path}")
        else:
            # Look for iteration checkpoints
            iter_checkpoints = [f for f in checkpoints if f.startswith('model_iter_')]
            if iter_checkpoints:
                # Extract iteration numbers and find latest
                def get_iter_num(filename):
                    match = re.search(r'model_iter_(\d+)\.pt', filename)
                    return int(match.group(1)) if match else 0
                
                iter_checkpoints.sort(key=get_iter_num, reverse=True)
                resume_path = os.path.join(checkpoint_dir, iter_checkpoints[0])
                print(f"Found iteration checkpoint: {resume_path}")
    
    # Create trainer and start training
    trainer = Trainer(
        checkpoint_dir=checkpoint_dir,
        log_dir="runs",
        device=device,
        use_amp=(device == 'cuda'),
        use_parallel=True
    )
    
    trainer.train(
        num_iterations=TRAINING_ITERATIONS,
        games_per_iteration=SELF_PLAY_GAMES,
        simulations_per_move=ALPHA_MCTS_SIMULATIONS,
        resume_from=resume_path
    )
    
    # Final evaluation
    print("\n" + "="*50)
    print("Final Evaluation...")
    evaluate_against_random(trainer.network, num_games=5, simulations=200)
