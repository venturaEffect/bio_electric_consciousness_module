import torch
import torch.nn as nn
from bcm.core.bioelectric_core import BioelectricState

class SimplifiedBioelectricCore(nn.Module):
    """
    A simplified version of the BCM core for web visualization.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.resting_potential = config.get('resting_potential', -0.2)
        self.gap_junction_strength = config.get('gap_junction_strength', 0.5)
        self.field_dimension = config.get('field_dimension', 10)
        
    def forward(self, state):
        """Process a state through simplified bioelectric dynamics."""
        # Extract voltage and ensure it's the right shape
        if len(state.voltage_potential.shape) != 2:
            voltage = state.voltage_potential.reshape(self.field_dimension, self.field_dimension)
        else:
            voltage = state.voltage_potential
            
        # Create a copy so we don't modify the original
        new_voltage = voltage.clone()
        
        # Apply gap junction diffusion (simplified)
        kernel = torch.tensor([
            [0.05, 0.1, 0.05],
            [0.1, 0.4, 0.1],
            [0.05, 0.1, 0.05]
        ], device=voltage.device)
        
        # Apply convolution manually (simplified diffusion)
        for i in range(1, voltage.shape[0] - 1):
            for j in range(1, voltage.shape[1] - 1):
                # Apply kernel to neighborhood
                new_voltage[i, j] = torch.sum(
                    voltage[i-1:i+2, j-1:j+2] * kernel
                )
        
        # Apply homeostatic drive toward resting potential
        homeostatic_drive = self.resting_potential - new_voltage
        new_voltage += 0.1 * homeostatic_drive
        
        # Apply non-linear activation (similar to voltage-gated channels)
        new_voltage = torch.tanh(new_voltage)
        
        # Create updated ion gradients (simplified)
        new_ion_gradients = {}
        for ion_name, gradient in state.ion_gradients.items():
            if ion_name == 'sodium':
                # Sodium follows voltage
                new_grad = torch.sigmoid(new_voltage) 
            elif ion_name == 'potassium':
                # Potassium is inverse to voltage
                new_grad = 1 - torch.sigmoid(new_voltage)
            else:
                # Others change more slowly
                new_grad = gradient * 0.9 + torch.sigmoid(new_voltage) * 0.1
            new_ion_gradients[ion_name] = new_grad
        
        # Update morphological state (integrate voltage information)
        if len(state.morphological_state.shape) != 1:
            morphological_state = state.morphological_state.flatten()
        else:
            morphological_state = state.morphological_state
        
        # Reshape voltage to update morphology
        flat_voltage = new_voltage.flatten()
        n = min(len(morphological_state), len(flat_voltage))
        
        # Simple integration of voltage into morphological state
        new_morphological_state = morphological_state.clone()
        new_morphological_state[:n] = morphological_state[:n] * 0.95 + flat_voltage[:n] * 0.05
        
        # Create new state
        new_state = BioelectricState(
            voltage_potential=new_voltage,
            ion_gradients=new_ion_gradients,
            gap_junction_states=state.gap_junction_states,
            morphological_state=new_morphological_state
        )
        
        return new_state