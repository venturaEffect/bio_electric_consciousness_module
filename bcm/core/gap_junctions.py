"""
Implementation of gap junction networks for cell-to-cell communication.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional

class GapJunctionNetwork(nn.Module):
    """
    Models gap junction communication between cells in a bioelectric network.
    
    Gap junctions are specialized intercellular connections that directly connect
    the cytoplasm of two cells, allowing various molecules, ions, and electrical impulses
    to pass through a regulated gate between cells.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        field_dim = config['field_dimension']
        
        # Gap junction connectivity matrix (learned)
        self.connectivity = nn.Parameter(
            torch.randn(field_dim, field_dim) * 0.01
        )
        
        # Attention-based gap junction mechanism
        self.gap_junction_attention = nn.MultiheadAttention(
            embed_dim=field_dim,
            num_heads=config.get('gap_junction_heads', 4),
            dropout=config.get('gap_junction_dropout', 0.1)
        )
        
    def forward(self, voltage_potential: torch.Tensor) -> torch.Tensor:
        """
        Process bioelectric signal through gap junctions
        
        Args:
            voltage_potential: Current voltage potential across the field
            
        Returns:
            Updated field after gap junction communication
        """
        # Reshape for attention mechanism
        field_reshaped = voltage_potential.unsqueeze(0).unsqueeze(0)
        
        # Apply attention-based communication
        field_communicated, _ = self.gap_junction_attention(
            field_reshaped, field_reshaped, field_reshaped
        )
        
        # Apply connectivity matrix (simulates physical connections between cells)
        connectivity_normalized = torch.sigmoid(self.connectivity)  # Normalize to 0-1
        field_after_gap = torch.matmul(connectivity_normalized, voltage_potential.unsqueeze(-1)).squeeze(-1)
        
        # Combine attention-based and physical gap junction effects
        field_communicated = field_communicated.squeeze(0).squeeze(0)
        combined_field = 0.7 * field_communicated + 0.3 * field_after_gap
        
        return combined_field