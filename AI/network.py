"""
AlphaZero Neural Network for Quoridor
ResNet-style architecture with policy and value heads.
"""
from __future__ import annotations
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import (
    GRID_COUNT, INPUT_CHANNELS, ACTION_SPACE_SIZE,
    NUM_RES_BLOCKS, NUM_FILTERS
)


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and skip connection.
    """
    
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = F.relu(out)
        
        return out


class QuoridorNet(nn.Module):
    """
    AlphaZero-style neural network for Quoridor.
    
    Architecture:
    - Input: (batch, 7, 9, 9) - encoded game state
    - Initial convolution to expand channels
    - Stack of residual blocks
    - Two heads:
      - Policy head: outputs (batch, 209) action probabilities
      - Value head: outputs (batch, 1) position evaluation
    """
    
    def __init__(
        self, 
        input_channels: int = INPUT_CHANNELS,
        num_filters: int = NUM_FILTERS,
        num_res_blocks: int = NUM_RES_BLOCKS,
        action_size: int = ACTION_SPACE_SIZE,
        board_size: int = GRID_COUNT
    ):
        super().__init__()
        
        self.board_size = board_size
        self.action_size = action_size
        self._device = None  # Cached device
        
        # Initial convolution
        self.conv_input = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * board_size * board_size, action_size)
        
        # Value head
        self.value_conv = nn.Conv2d(num_filters, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, channels, 9, 9)
        
        Returns:
            policy_logits: (batch, 209) raw logits for each action
            value: (batch, 1) evaluation in range [-1, 1]
        """
        # Initial convolution
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = F.relu(x)
        
        # Residual tower
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Policy head
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy_logits = self.policy_fc(policy)
        
        # Value head
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy_logits, value
    
    @property
    def device(self) -> torch.device:
        """Get cached device of the model."""
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device
    
    def predict(
        self, 
        state: np.ndarray, 
        legal_mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, float]:
        """
        Get policy and value prediction for a single state.
        
        Args:
            state: Encoded state of shape (channels, 9, 9)
            legal_mask: Optional boolean mask of legal actions
        
        Returns:
            policy: Probability distribution over actions (209,)
            value: Position evaluation scalar
        """
        with torch.inference_mode():
            # Add batch dimension and convert to tensor - use cached device
            x = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
            
            policy_logits, value = self.forward(x)
            
            # Apply legal action mask on CPU
            policy_logits = policy_logits[0].cpu().numpy()
            
            if legal_mask is not None:
                policy_logits[~legal_mask] = -np.inf
            
            # Convert to probabilities
            policy = self._softmax(policy_logits)
            
            return policy, value.item()
    
    def predict_batch(
        self, 
        states: np.ndarray, 
        legal_masks: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get policy and value predictions for a batch of states.
        
        Args:
            states: Batch of encoded states (batch, channels, 9, 9)
            legal_masks: Optional batch of legal action masks (batch, 209)
        
        Returns:
            policies: Batch of probability distributions (batch, 209)
            values: Batch of evaluations (batch,)
        """
        with torch.inference_mode():
            x = torch.from_numpy(states.astype(np.float32)).to(self.device, non_blocking=True)
            
            policy_logits, values = self.forward(x)
            
            # Apply legal mask before softmax (on GPU if available)
            if legal_masks is not None:
                mask_tensor = torch.from_numpy(legal_masks).to(self.device, non_blocking=True)
                policy_logits = policy_logits.masked_fill(~mask_tensor, float('-inf'))
            
            # Softmax on GPU (much faster than numpy)
            policies = torch.softmax(policy_logits, dim=1)
            
            # Move to CPU for return
            policies = policies.cpu().numpy()
            values = values.cpu().numpy().flatten()
            
            return policies, values
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        # Handle -inf values
        finite_mask = np.isfinite(x)
        if not np.any(finite_mask):
            # All masked - return uniform over all (shouldn't happen)
            return np.ones_like(x) / len(x)
        
        x_max = np.max(x[finite_mask])
        exp_x = np.exp(x - x_max)
        exp_x[~finite_mask] = 0
        return exp_x / np.sum(exp_x)
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save model weights to file."""
        torch.save({
            'state_dict': self.state_dict(),
            'board_size': self.board_size,
            'action_size': self.action_size,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load model weights from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'])
        print(f"Model loaded from {filepath}")


def create_network(device: str = 'cpu') -> QuoridorNet:
    """
    Create and initialize the neural network.
    
    Args:
        device: 'cpu' or 'cuda'
    
    Returns:
        Initialized QuoridorNet
    """
    net = QuoridorNet()
    
    # Initialize weights
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    if device == 'cuda' and torch.cuda.is_available():
        net = net.cuda()
        net._device = torch.device('cuda')  # Set cached device
    else:
        net._device = torch.device('cpu')
    
    return net
