"""
Utility functions for measuring and analyzing bioelectric states.
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from bcm.core.bioelectric_core import BioelectricState

def calculate_pattern_stability(state: BioelectricState, window: int = 5) -> float:
    """
    Calculate the stability of bioelectric patterns over time.
    
    Args:
        state: Current bioelectric state
        window: Number of past states to consider
        
    Returns:
        Stability score (0-1)
    """
    # This is a placeholder - in a real implementation, you would
    # track state history and calculate true stability over time
    
    # For now, use the inverse of the standard deviation as a simple
    # proxy for stability (more uniform = more stable)
    voltage_std = state.voltage_potential.std().item()
    stability = 1.0 / (1.0 + voltage_std * 5.0)  # Scale to get values closer to 0-1
    
    # Ensure value is in 0-1 range
    stability = max(0.0, min(1.0, stability))
    
    return stability

def calculate_bioelectric_entropy(state: BioelectricState) -> float:
    """
    Calculate information entropy in the bioelectric field.
    
    Args:
        state: Current bioelectric state
        
    Returns:
        Entropy value
    """
    # Convert voltage potentials to probability distribution
    voltage = state.voltage_potential
    
    # Shift to ensure all positive values
    voltage_shifted = voltage - voltage.min() + 1e-8
    
    # Normalize to create a probability distribution
    prob_dist = voltage_shifted / voltage_shifted.sum()
    
    # Calculate entropy
    entropy = -torch.sum(prob_dist * torch.log2(prob_dist))
    
    return entropy.item()

def calculate_spatial_coherence(state: BioelectricState) -> float:
    """
    Calculate spatial coherence of the bioelectric pattern.
    
    Args:
        state: Current bioelectric state
        
    Returns:
        Coherence score (0-1)
    """
    voltage = state.voltage_potential
    
    # Calculate gradient of voltage field
    if len(voltage.shape) > 1:
        # For 2D or higher fields, calculate spatial gradient
        # This is a simplification - adapt based on your actual voltage field shape
        dx = torch.diff(voltage, dim=0)
        gradient_magnitude = torch.mean(torch.abs(dx))
    else:
        # For 1D fields
        gradient_magnitude = torch.mean(torch.abs(torch.diff(voltage)))
    
    # Convert gradient to coherence (high gradient = low coherence)
    coherence = torch.exp(-gradient_magnitude * 3.0).item()
    
    return coherence

def calculate_goal_alignment(state: BioelectricState, goal_state: torch.Tensor) -> float:
    """
    Calculate alignment between current state and goal state.
    
    Args:
        state: Current bioelectric state
        goal_state: Desired goal state
        
    Returns:
        Alignment score (0-1)
    """
    # Calculate cosine similarity between current voltage and goal state
    current = state.voltage_potential
    similarity = torch.nn.functional.cosine_similarity(
        current.view(1, -1),
        goal_state.view(1, -1)
    )
    
    # Convert from -1,1 to 0,1 range
    alignment = (similarity.item() + 1) / 2
    
    return alignment

def summarize_bioelectric_state(state: BioelectricState) -> Dict[str, Any]:
    """
    Create a comprehensive summary of bioelectric state metrics.
    
    Args:
        state: Current bioelectric state
        
    Returns:
        Dictionary of metrics
    """
    return {
        'mean_voltage': state.voltage_potential.mean().item(),
        'voltage_std': state.voltage_potential.std().item(),
        'pattern_stability': calculate_pattern_stability(state),
        'bioelectric_entropy': calculate_bioelectric_entropy(state),
        'spatial_coherence': calculate_spatial_coherence(state),
        'active_regions': (state.voltage_potential > 0.5).sum().item(),
        'peak_voltage': state.voltage_potential.max().item()
    }