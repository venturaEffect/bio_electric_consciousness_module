"""
Implementation of bioelectric pattern formation mechanisms.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional

class BioelectricPatternFormation(nn.Module):
    """
    Models the formation of bioelectric patterns, inspired by
    pattern formation processes observed in developmental biology.
    
    This implements a reaction-diffusion-like system that creates
    self-organizing patterns from simple local rules.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        field_dim = config['field_dimension']
        
        # Pattern activation parameters
        self.activation_weights = nn.Parameter(torch.randn(field_dim, field_dim) * 0.01)
        self.inhibition_weights = nn.Parameter(torch.randn(field_dim, field_dim) * 0.01)
        
        # Pattern stability parameters
        self.stability_preference = config['morphology'].get('stability_preference', 0.7)
        self.plasticity = config['morphology'].get('plasticity', 0.3)
        
    def update_pattern(self, 
                      voltage_potential: torch.Tensor, 
                      morphological_state: torch.Tensor) -> torch.Tensor:
        """
        Update morphological pattern based on current bioelectric state
        
        Args:
            voltage_potential: Current voltage distribution
            morphological_state: Current morphological configuration
            
        Returns:
            Updated morphological pattern
        """
        # Calculate activator influence
        activation = torch.matmul(torch.sigmoid(self.activation_weights), voltage_potential.unsqueeze(-1)).squeeze(-1)
        
        # Calculate inhibitor influence
        inhibition = torch.matmul(torch.sigmoid(self.inhibition_weights), voltage_potential.unsqueeze(-1)).squeeze(-1)
        
        # Apply reaction-diffusion dynamics
        pattern_change = activation - inhibition
        
        # Update pattern with stability/plasticity balance
        new_pattern = (
            self.stability_preference * morphological_state +
            self.plasticity * pattern_change
        )
        
        # Normalize to prevent unbounded growth
        new_pattern = torch.tanh(new_pattern)
        
        return new_pattern
    
    def get_pattern_metrics(self, pattern: torch.Tensor) -> Dict:
        """
        Calculate metrics that characterize the bioelectric pattern
        
        Args:
            pattern: Current morphological pattern
            
        Returns:
            Dictionary of pattern metrics
        """
        # Calculate pattern complexity
        fft_result = torch.fft.rfft(pattern)
        power_spectrum = torch.abs(fft_result)**2
        
        # Pattern symmetry (simplified)
        half_len = pattern.shape[0] // 2
        if half_len > 0:
            first_half = pattern[:half_len]
            second_half = pattern[half_len:2*half_len]
            if second_half.shape[0] < first_half.shape[0]:  # Handle odd lengths
                second_half = torch.cat([second_half, torch.zeros(1)])
            symmetry = 1.0 - torch.mean(torch.abs(first_half - torch.flip(second_half, [0]))).item()
        else:
            symmetry = 1.0
        
        # Pattern energy
        energy = torch.sum(pattern**2).item()
        
        # Pattern sparsity
        sparsity = torch.sum(pattern.abs() < 0.1).item() / pattern.shape[0]
        
        return {
            'symmetry': symmetry,
            'energy': energy,
            'sparsity': sparsity,
            'dominant_frequency': torch.argmax(power_spectrum).item()
        }