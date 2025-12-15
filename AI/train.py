"""
AlphaZero Training Pipeline for Quoridor
Includes self-play data generation, replay buffer, and training loop.
"""
from __future__ import annotations
import os
import sys

# Add parent directory to path for imports when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from game import GameState
from AI.network import QuoridorNet, create_network
from AI.alpha_mcts import AlphaMCTS
from AI.encoder import encode_state
from AI.action_utils import action_to_index, get_legal_action_mask

from constants import (
    SELF_PLAY_GAMES, REPLAY_BUFFER_SIZE, BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY, TRAINING_ITERATIONS,
    CHECKPOINT_INTERVAL, TEMP_THRESHOLD, TEMP_INIT, TEMP_FINAL,
    ALPHA_MCTS_SIMULATIONS, ACTION_SPACE_SIZE
)


@dataclass
class TrainingExample:
    """A single training example from self-play."""
    state: np.ndarray          # Encoded board state
    policy: np.ndarray         # MCTS visit count distribution
    value: float               # Game outcome from this player's perspective


class ReplayBuffer:
    """
    Fixed-size buffer to store training examples from self-play.
    """
    
    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)
    
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


def self_play_game(
    network: QuoridorNet,
    num_simulations: int = ALPHA_MCTS_SIMULATIONS,
    temp_threshold: int = TEMP_THRESHOLD,
    verbose: bool = False
) -> list[TrainingExample]:
    """
    Play one game of self-play and collect training data.
    
    Args:
        network: Neural network for MCTS
        num_simulations: MCTS simulations per move
        temp_threshold: Move number after which temperature drops
        verbose: Print game progress
    
    Returns:
        List of training examples from the game
    """
    mcts = AlphaMCTS(
        network=network,
        num_simulations=num_simulations,
        add_noise=True  # Exploration noise during self-play
    )
    
    game_state = GameState()
    game_history = []  # (state, policy, current_player)
    move_count = 0
    
    while not game_state.is_terminal():
        # Temperature schedule: high early for exploration, low later
        if move_count < temp_threshold:
            temperature = TEMP_INIT
        else:
            temperature = TEMP_FINAL
        
        # Get MCTS policy
        action_probs, _ = mcts.get_action_probs(game_state, temperature=temperature)
        
        # Store training data (before move)
        encoded_state = encode_state(game_state)
        current_player = game_state.current_player
        game_history.append((encoded_state, action_probs, current_player))
        
        # Select and apply action
        if temperature == 0:
            action_idx = np.argmax(action_probs)
        else:
            action_idx = np.random.choice(len(action_probs), p=action_probs)
        
        from AI.action_utils import index_to_action
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
        if winner is None:
            value = 0.0  # Draw (shouldn't happen in Quoridor)
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
    
    return training_examples


def generate_self_play_data(
    network: QuoridorNet,
    num_games: int = SELF_PLAY_GAMES,
    num_simulations: int = ALPHA_MCTS_SIMULATIONS,
    verbose: bool = True
) -> list[TrainingExample]:
    """
    Generate training data through self-play.
    
    Args:
        network: Neural network for MCTS
        num_games: Number of self-play games
        num_simulations: MCTS simulations per move
        verbose: Print progress
    
    Returns:
        List of all training examples from all games
    """
    all_examples = []
    
    for game_idx in range(num_games):
        if verbose:
            print(f"\nSelf-play game {game_idx + 1}/{num_games}")
        
        examples = self_play_game(
            network=network,
            num_simulations=num_simulations,
            verbose=False
        )
        all_examples.extend(examples)
        
        if verbose:
            print(f"  Generated {len(examples)} examples (total: {len(all_examples)})")
    
    return all_examples


def train_network(
    network: QuoridorNet,
    examples: list[TrainingExample],
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    epochs: int = 10,
    verbose: bool = True
) -> dict:
    """
    Train the neural network on collected examples.
    
    Args:
        network: Neural network to train
        examples: Training examples
        batch_size: Mini-batch size
        learning_rate: Learning rate
        weight_decay: L2 regularization
        epochs: Training epochs over the data
        verbose: Print training progress
    
    Returns:
        Dictionary with training metrics
    """
    network.train()
    device = next(network.parameters()).device
    
    # Create data loader
    dataset = TrainingDataset(examples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = optim.Adam(
        network.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Loss functions
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    
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
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)
            
            # Forward pass
            policy_logits, pred_values = network(states)
            
            # Policy loss (cross-entropy with MCTS policy as target)
            policy_loss = -torch.mean(
                torch.sum(policies * torch.log_softmax(policy_logits, dim=1), dim=1)
            )
            
            # Value loss (MSE)
            value_loss = value_loss_fn(pred_values, values)
            
            # Total loss
            total_loss = policy_loss + value_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_total_loss += total_loss.item()
            num_batches += 1
        
        # Average losses for epoch
        avg_policy = epoch_policy_loss / num_batches
        avg_value = epoch_value_loss / num_batches
        avg_total = epoch_total_loss / num_batches
        
        metrics['policy_loss'].append(avg_policy)
        metrics['value_loss'].append(avg_value)
        metrics['total_loss'].append(avg_total)
        
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"Policy={avg_policy:.4f}, Value={avg_value:.4f}, Total={avg_total:.4f}")
    
    return metrics


def training_loop(
    num_iterations: int = TRAINING_ITERATIONS,
    games_per_iteration: int = SELF_PLAY_GAMES,
    simulations_per_move: int = ALPHA_MCTS_SIMULATIONS,
    checkpoint_dir: str = "checkpoints",
    resume_from: Optional[str] = None,
    device: str = 'cpu'
) -> QuoridorNet:
    """
    Main AlphaZero training loop.
    
    Args:
        num_iterations: Number of training iterations
        games_per_iteration: Self-play games per iteration
        simulations_per_move: MCTS simulations per move
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from
        device: 'cpu' or 'cuda'
    
    Returns:
        Trained neural network
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize or load network
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        network = create_network(device)
        network.load_checkpoint(resume_from)
    else:
        print("Starting fresh training")
        network = create_network(device)
    
    # Replay buffer
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    
    for iteration in range(1, num_iterations + 1):
        print(f"\n{'='*50}")
        print(f"Training Iteration {iteration}/{num_iterations}")
        print(f"{'='*50}")
        
        # Phase 1: Self-play
        print("\n--- Self-Play Phase ---")
        start_time = time.time()
        
        examples = generate_self_play_data(
            network=network,
            num_games=games_per_iteration,
            num_simulations=simulations_per_move,
            verbose=True
        )
        
        replay_buffer.add_batch(examples)
        
        self_play_time = time.time() - start_time
        print(f"Self-play completed in {self_play_time:.1f}s")
        print(f"Replay buffer size: {len(replay_buffer)}")
        
        # Phase 2: Training
        print("\n--- Training Phase ---")
        start_time = time.time()
        
        # Sample from replay buffer
        training_examples = replay_buffer.sample(min(len(replay_buffer), BATCH_SIZE * 100))
        
        metrics = train_network(
            network=network,
            examples=training_examples,
            epochs=10,
            verbose=True
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.1f}s")
        
        # Save checkpoint
        if iteration % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_iter_{iteration}.pt")
            network.save_checkpoint(checkpoint_path)
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, "model_final.pt")
    network.save_checkpoint(final_path)
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    
    return network


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


if __name__ == "__main__":
    # Quick training test
    print("Starting AlphaZero training for Quoridor...")
    print("Note: This is computationally intensive!")
    print("Consider reducing SELF_PLAY_GAMES and ALPHA_MCTS_SIMULATIONS for testing.\n")
    
    # Use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Train with reduced settings for testing
    network = training_loop(
        num_iterations=5,  # Reduced for testing
        games_per_iteration=10,  # Reduced for testing
        simulations_per_move=50,  # Reduced for testing
        checkpoint_dir="checkpoints",
        device=device
    )
    
    # Evaluate
    print("\n" + "="*50)
    print("Evaluating trained model...")
    evaluate_against_random(network, num_games=10, simulations=50)
