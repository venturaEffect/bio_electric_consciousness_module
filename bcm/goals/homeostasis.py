"""
Implementation of homeostatic mechanisms for bioelectric systems.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

class HomeostasisRegulator(nn.Module):
    """
    Models homeostatic regulation in bioelectric systems.
    
    Homeostasis is the tendency of biological systems to maintain
    stable internal states despite changing external conditions.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Target states (resting potentials)
        self.target_voltage = torch.tensor(config.get('resting_potential', 0.2))
        self.target_ranges = {
            'sodium': (0.05, 0.15),
            'potassium': (0.85, 0.95),
            'calcium': (0.15, 0.25)
        }
        
        # Homeostasis regulation network
        field_dim = config['field_dimension']
        self.homeostasis_network = nn.Sequential(
            nn.Linear(field_dim, field_dim),
            nn.Sigmoid(),
            nn.Linear(field_dim, field_dim)
        )
        
    def calculate_homeostatic_drive(self, 
                                   voltage_potential: torch.Tensor, 
                                   ion_gradients: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate homeostatic drive to restore stable state
        
        Args:
            voltage_potential: Current voltage potential
            ion_gradients: Current ion gradients
            
        Returns:
            Homeostatic correction signal
        """
        # Calculate voltage error
        voltage_error = self.target_voltage - voltage_potential
        
        # Calculate ion gradient errors
        ion_errors = {}
        for ion, gradient in ion_gradients.items():
            if ion in self.target_ranges:
                target_min, target_max = self.target_ranges[ion]
                target_mid = (target_min + target_max) / 2
                
                # Error is stronger if outside target range
                error = target_mid - gradient
                # Amplify error if outside range
                outside_range = (gradient < target_min) | (gradient > target_max)
                error = error * (1.0 + outside_range.float())
                
                ion_errors[ion] = error
        
        # Combine errors
        combined_error = voltage_error
        for error in ion_errors.values():
            combined_error = combined_error + error * 0.2  # Ion errors have lower weight
        
        # Calculate homeostatic response through network
        homeostatic_response = self.homeostasis_network(combined_error)
        
        return homeostatic_response
    
    def apply_homeostatic_correction(self,
                                    voltage_potential: torch.Tensor,
                                    ion_gradients:"""
Implementation of homeostatic mechanisms for bioelectric systems.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

class HomeostasisRegulator(nn.Module):
    """
    Models homeostatic regulation in bioelectric systems.
    
    Homeostasis is the tendency of biological systems to maintain
    stable internal states despite changing external conditions.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Target states (resting potentials)
        self.target_voltage = torch.tensor(config.get('resting_potential', 0.2))
        self.target_ranges = {
            'sodium': (0.05, 0.15),
            'potassium': (0.85, 0.95),
            'calcium': (0.15, 0.25)
        }
        
        # Homeostasis regulation network
        field_dim = config['field_dimension']
        self.homeostasis_network = nn.Sequential(
            nn.Linear(field_dim, field_dim),
            nn.Sigmoid(),
            nn.Linear(field_dim, field_dim)
        )
        
    def calculate_homeostatic_drive(self, 
                                   voltage_potential: torch.Tensor, 
                                   ion_gradients: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate homeostatic drive to restore stable state
        
        Args:
            voltage_potential: Current voltage potential
            ion_gradients: Current ion gradients
            
        Returns:
            Homeostatic correction signal
        """
        # Calculate voltage error
        voltage_error = self.target_voltage - voltage_potential
        
        # Calculate ion gradient errors
        ion_errors = {}
        for ion, gradient in ion_gradients.items():
            if ion in self.target_ranges:
                target_min, target_max = self.target_ranges[ion]
                target_mid = (target_min + target_max) / 2
                
                # Error is stronger if outside target range
                error = target_mid - gradient
                # Amplify error if outside range
                outside_range = (gradient < target_min) | (gradient > target_max)
                error = error * (1.0 + outside_range.float())
                
                ion_errors[ion] = error
        
        # Combine errors
        combined_error = voltage_error
        for error in ion_errors.values():
            combined_error = combined_error + error * 0.2  # Ion errors have lower weight
        
        # Calculate homeostatic response through network
        homeostatic_response = self.homeostasis_network(combined_error)
        
        return homeostatic_response
    
    def apply_homeostatic_correction(self,
                                    voltage_potential: torch.Tensor,
                                    ion_gradients: