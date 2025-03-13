"""
Implementation of stigmergy mechanisms for indirect coordination.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np

class StigmergyEnvironment:
    """
    Models environmental communication via stigmergy.
    
    Stigmergy is a mechanism of indirect coordination between agents through
    the environment. Agents modify the environment, and these modifications
    influence the behavior of other agents, forming a feedback loop.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.field_dim = config.get('environment_dimension', 64)
        
        # Initialize environment state
        self.reset()
        
    def reset(self):
        """Reset the environment to its initial state"""
        self.environment_state = torch.zeros(self.field_dim)
        self.signal_strength = torch.zeros(self.field_dim)
        self.signal_decay = self.config.get('signal_decay_rate', 0.95)
    
    def update(self, signals: List[torch.Tensor], positions: Optional[List[int]] = None):
        """
        Update environment based on signals from cells
        
        Args:
            signals: List of signals from cells
            positions: Optional positions of cells in the environment
        """
        # Decay existing signals
        self.signal_strength = self.signal_strength * self.signal_decay
        
        # Add new signals
        for i, signal in enumerate(signals):
            if positions and i < len(positions):
                # Place signal at specific position
                pos = positions[i]
                if 0 <= pos < self.field_dim:
                    # Diffuse signal around position
                    for j in range(max(0, pos-5), min(self.field_dim, pos+6)):
                        distance = abs(j - pos)
                        diffusion = max(0, 1.0 - distance * 0.2) 
                        self.signal_strength[j] += signal.mean().item() * diffusion
            else:
                # Distributed signal across environment
                self.signal_strength += signal.mean().item() / self.field_dim
        
        # Update environment state based on signal strength
        self.environment_state = torch.tanh(self.environment_state + self.signal_strength)
    
    def get_local_state(self, position: int, radius: int = 5) -> torch.Tensor:
        """
        Get local environment state around a position
        
        Args:
            position: Center position
            radius: Radius around position
            
        Returns:
            Local environment state
        """
        start = max(0, position - radius)
        end = min(self.field_dim, position + radius + 1)
        
        local_state = self.environment_state[start:end]
        
        # Pad if necessary to maintain consistent size
        if local_state.shape[0] < 2 * radius + 1:
            padding = 2 * radius + 1 - local_state.shape[0]
            local_state = torch.nn.functional.pad(local_state, (0, padding))
        
        return local_state
    
    def get_global_state(self) -> torch.Tensor:
        """Get full environment state"""
        return self.environment_state.clone()