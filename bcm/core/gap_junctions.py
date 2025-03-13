"""
Implementation of gap junctions for cell-to-cell communication.
Gap junctions allow direct electrical and metabolic coupling between cells,
enabling collective information processing in cell networks.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any

class GapJunctionNetwork(nn.Module):
    """
    Models communication between cells via gap junctions.
    Gap junctions are channels that directly connect the cytoplasm of adjacent cells,
    allowing electrical signals and small molecules to pass between them.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.num_cells = config.get('num_cells', 16)
        self.dimension = config.get('dimension', 64)
        
        # Learnable adjacency matrix representing gap junction connections
        # Initialize with small random values
        self.adjacency = nn.Parameter(torch.rand(self.num_cells, self.num_cells) * 0.1)
        
        # Gating mechanism that can dynamically open/close gap junctions
        self.gate_network = nn.Sequential(
            nn.Linear(self.dimension * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Transformation for exchanged signals
        self.signal_transform = nn.Linear(self.dimension, self.dimension)
        
    def forward(self, cell_states: torch.Tensor) -> torch.Tensor:
        """
        Process communication between cells via gap junctions.
        
        Args:
            cell_states: Tensor of shape [num_cells, dimension] representing states of each cell
            
        Returns:
            Updated cell states after gap junction communication
        """
        batch_size = cell_states.shape[0]
        
        # Create symmetric adjacency matrix with no self-connections
        sym_adj = (self.adjacency + self.adjacency.t()) / 2
        sym_adj = sym_adj * (1 - torch.eye(self.num_cells, device=sym_adj.device))
        
        # Apply softmax to normalize connection strengths
        normalized_adj = torch.softmax(sym_adj, dim=1)
        
        # Initialize updated states
        updated_states = cell_states.clone()
        
        # For each cell, compute the effect of gap junctions
        for i in range(self.num_cells):
            # Cell's own state
            own_state = cell_states[i:i+1]
            
            # Calculate gap junction effects from all other cells
            for j in range(self.num_cells):
                if i != j:
                    # Other cell's state
                    other_state = cell_states[j:j+1]
                    
                    # Calculate gap junction gating
                    combined = torch.cat([own_state, other_state], dim=1)
                    gate_value = self.gate_network(combined)
                    
                    # Apply connection strength from adjacency matrix
                    connection_strength = normalized_adj[i, j]
                    
                    # Calculate signal that passes through the gap junction
                    transferred_signal = self.signal_transform(other_state)
                    
                    # Update the cell's state with signals from this neighbor
                    # scaled by gate value and connection strength
                    updated_states[i] += gate_value * connection_strength * transferred_signal.squeeze(0)
        
        return updated_states
    
    def evolve_connectivity(self, reward_signal: torch.Tensor) -> None:
        """
        Evolve gap junction connectivity based on feedback/reward.
        This simulates the biological phenomenon where gap junctions
        can be reinforced or weakened based on signaling activity.
        
        Args:
            reward_signal: Signal indicating benefit of current connectivity pattern
        """
        # Simple update rule - strengthen connections that received positive reward
        with torch.no_grad():
            # Apply small changes to adjacency matrix
            delta = torch.randn_like(self.adjacency) * 0.01 * reward_signal
            self.adjacency.data += delta
            
            # Optional: Apply constraints (e.g., keep values in reasonable range)
            self.adjacency.data.clamp_(0.0, 1.0)