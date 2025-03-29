import torch
import numpy as np

class SimpleBioelectricModel(torch.nn.Module):
    """A simplified model for testing visualization."""
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        
    def forward(self, state):
        """Basic diffusion model."""
        # Simple diffusion by blurring the voltage field
        voltage = state.voltage_potential.clone()
        
        # Apply a simple convolution for diffusion
        padded = torch.nn.functional.pad(voltage, (1, 1, 1, 1), mode='replicate')
        kernel = torch.tensor([[0.05, 0.1, 0.05], 
                               [0.1, 0.4, 0.1], 
                               [0.05, 0.1, 0.05]], device=voltage.device)
        
        # Apply kernel manually for clear understanding
        new_voltage = voltage.clone()
        for i in range(voltage.shape[0]):
            for j in range(voltage.shape[1]):
                # Extract 3x3 neighborhood from padded tensor
                neighborhood = padded[i:i+3, j:j+3]
                # Apply kernel
                new_voltage[i, j] = torch.sum(neighborhood * kernel)
        
        # Add some noise to create patterns
        noise = torch.randn_like(voltage) * 0.02
        new_voltage = new_voltage + noise
        
        # Update ion gradients based on voltage
        new_ion_gradients = {}
        for ion_name, gradient in state.ion_gradients.items():
            if ion_name == 'sodium':
                # Sodium follows voltage
                new_ion_gradients[ion_name] = torch.sigmoid(new_voltage)
            elif ion_name == 'potassium':
                # Potassium is inverse to voltage
                new_ion_gradients[ion_name] = 1.0 - torch.sigmoid(new_voltage)
            else:
                # Others change more slowly
                new_ion_gradients[ion_name] = gradient * 0.95 + torch.sigmoid(new_voltage) * 0.05
                
        # Update morphological state - integrate voltage
        if len(state.morphological_state.shape) == 1:
            # Convert to 2D for visualization
            field_dim = int(np.sqrt(state.morphological_state.shape[0]))
            morph_2d = state.morphological_state.reshape(field_dim, field_dim)
        else:
            morph_2d = state.morphological_state
            
        # Update morphology more slowly
        new_morph = morph_2d * 0.95 + new_voltage * 0.05
        
        # Flatten for storage
        new_morph_flat = new_morph.flatten()
        
        state.voltage_potential = new_voltage
        state.ion_gradients = new_ion_gradients
        state.morphological_state = new_morph_flat
        
        return state