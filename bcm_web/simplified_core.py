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
        # Extract components
        voltage = state.voltage_potential
        ion_gradients = state.ion_gradients
        
        # Simple voltage diffusion through gap junctions
        voltage_laplacian = self._compute_laplacian(voltage)
        voltage_diffusion = self.gap_junction_strength * voltage_laplacian
        
        # Simple ion channel effects (depolarization/repolarization)
        sodium_effect = 0.1 * ion_gradients['sodium']
        potassium_effect = -0.05 * ion_gradients['potassium']
        calcium_effect = 0.08 * ion_gradients['calcium']
        
        # Update voltage
        voltage_update = voltage_diffusion + sodium_effect + potassium_effect + calcium_effect
        new_voltage = voltage + 0.1 * voltage_update
        
        # Apply non-linearity to keep values in reasonable range
        new_voltage = torch.tanh(new_voltage)
        
        # Simple updates to ion gradients based on voltage
        new_sodium = ion_gradients['sodium'] * 0.95 + 0.05 * torch.relu(new_voltage)
        new_potassium = ion_gradients['potassium'] * 0.95 + 0.05 * torch.relu(-new_voltage)
        new_calcium = ion_gradients['calcium'] * 0.95 + 0.05 * torch.abs(new_voltage)
        
        # Create updated state
        new_state = BioelectricState(
            voltage_potential=new_voltage,
            ion_gradients={
                'sodium': new_sodium,
                'potassium': new_potassium,
                'calcium': new_calcium
            },
            gap_junction_states=state.gap_junction_states,
            morphological_state=state.morphological_state
        )
        
        return new_state
    
    def _compute_laplacian(self, tensor):
        """Compute discrete Laplacian operator (for diffusion)."""
        # Get dimensions
        h, w = tensor.shape
        
        # Compute discrete Laplacian using convolution-like operations
        laplacian = torch.zeros_like(tensor)
        
        # Internal points
        for i in range(1, h-1):
            for j in range(1, w-1):
                laplacian[i, j] = (
                    tensor[i+1, j] + tensor[i-1, j] + 
                    tensor[i, j+1] + tensor[i, j-1] - 
                    4 * tensor[i, j]
                )
                
        # Handle boundary conditions (no-flux)
        # Top and bottom edges
        for j in range(1, w-1):
            laplacian[0, j] = 2 * tensor[1, j] + tensor[0, j+1] + tensor[0, j-1] - 4 * tensor[0, j]
            laplacian[h-1, j] = 2 * tensor[h-2, j] + tensor[h-1, j+1] + tensor[h-1, j-1] - 4 * tensor[h-1, j]
        
        # Left and right edges
        for i in range(1, h-1):
            laplacian[i, 0] = tensor[i+1, 0] + tensor[i-1, 0] + 2 * tensor[i, 1] - 4 * tensor[i, 0]
            laplacian[i, w-1] = tensor[i+1, w-1] + tensor[i-1, w-1] + 2 * tensor[i, w-2] - 4 * tensor[i, w-1]
        
        # Corners
        laplacian[0, 0] = 2 * tensor[1, 0] + 2 * tensor[0, 1] - 4 * tensor[0, 0]
        laplacian[0, w-1] = 2 * tensor[1, w-1] + 2 * tensor[0, w-2] - 4 * tensor[0, w-1]
        laplacian[h-1, 0] = 2 * tensor[h-2, 0] + 2 * tensor[h-1, 1] - 4 * tensor[h-1, 0]
        laplacian[h-1, w-1] = 2 * tensor[h-2, w-1] + 2 * tensor[h-1, w-2] - 4 * tensor[h-1, w-1]
        
        return laplacian