"""
Environment setup for bioelectric consciousness simulations.
"""
import torch
import numpy as np
from typing import Dict, Tuple, List, Optional, Any

class BioelectricEnvironment:
    """
    Defines the environment in which bioelectric systems operate.
    Controls external conditions like nutrient availability, 
    signal gradients, and environmental perturbations.
    """
    def __init__(self, 
                 size: Tuple[int, int],
                 boundary_condition: str = "periodic",
                 noise_level: float = 0.05):
        """
        Initialize the bioelectric environment.
        
        Args:
            size: Spatial dimensions of the environment (height, width)
            boundary_condition: Type of boundary condition ("periodic", "fixed", "reflective")
            noise_level: Amount of random noise in the environment
        """
        self.size = size
        self.boundary_condition = boundary_condition
        self.noise_level = noise_level
        self.initialize_environment()
    
    def initialize_environment(self):
        """Initialize environment variables."""
        # Base chemical gradients
        self.ion_gradients = {
            'sodium': self._create_gradient((0, 0), self.size, 1.0, 0.2),
            'potassium': self._create_gradient((self.size[0], self.size[1]), self.size, 0.8, 0.3),
            'calcium': self._create_gradient((self.size[0]//2, self.size[1]//2), self.size, 1.2, 0.5),
        }
        
    def _create_gradient(self, 
                         center: Tuple[int, int], 
                         size: Tuple[int, int], 
                         amplitude: float, 
                         decay: float) -> np.ndarray:
        """Create a radial gradient in the environment."""
        y, x = np.ogrid[:size[0], :size[1]]
        dist = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        gradient = amplitude * np.exp(-decay * dist)
        return gradient
    
    def apply_perturbation(self, 
                          location: Tuple[int, int], 
                          radius: int, 
                          strength: float) -> None:
        """
        Apply a localized perturbation to the environment.
        
        Args:
            location: Center point of perturbation (y, x)
            radius: Radius of the affected area
            strength: Strength of the perturbation
        """
        y, x = np.ogrid[:self.size[0], :self.size[1]]
        dist = np.sqrt((x - location[1])**2 + (y - location[0])**2)
        mask = dist <= radius
        
        # Perturb all ion gradients
        for ion in self.ion_gradients:
            self.ion_gradients[ion][mask] += strength
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Get the current state of the environment.
        
        Returns:
            Dict containing ion gradients and other environment variables
        """
        state = {
            'ion_gradients': self.ion_gradients,
            'noise': self.noise_level * np.random.randn(*self.size)
        }
        return state
    
    def update(self) -> None:
        """Update the environment for the next time step."""
        # Apply diffusion to ion gradients
        for ion in self.ion_gradients:
            # Add small random fluctuations
            self.ion_gradients[ion] += self.noise_level * np.random.randn(*self.size)
            
            # Handle boundary conditions
            if self.boundary_condition == "fixed":
                # Zero at boundaries
                self.ion_gradients[ion][0, :] = 0
                self.ion_gradients[ion][-1, :] = 0
                self.ion_gradients[ion][:, 0] = 0
                self.ion_gradients[ion][:, -1] = 0