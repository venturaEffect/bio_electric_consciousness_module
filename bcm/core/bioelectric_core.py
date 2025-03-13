"""
Core implementation of the bioelectric processing system.
Inspired by voltage-mediated information processing in cells as described
in the work of Michael Levin et al.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

@dataclass
class BioelectricState:
    """Tracks bioelectric state of a primitive conscious entity."""
    voltage_potential: torch.Tensor  # Cell membrane potentials
    ion_gradients: Dict[str, torch.Tensor]  # Different ion channel states
    gap_junction_states: torch.Tensor  # Connectivity between units
    morphological_state: torch.Tensor  # Physical configuration

class BioelectricConsciousnessCore(nn.Module):
    """
    Primitive consciousness core based on bioelectric signaling principles.
    Models bioelectric pattern formation and information processing in cells.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Core bioelectric components
        self.ion_channels = nn.ModuleDict({
            'sodium': self._create_ion_channel(config['sodium_channel']),
            'potassium': self._create_ion_channel(config['potassium_channel']),
            'calcium': self._create_ion_channel(config['calcium_channel'])
        })
        
        # Gap junction network (cell-to-cell communication)
        self.gap_junction_network = nn.Linear(
            config['field_dimension'], 
            config['field_dimension']
        )
        
        # Morphological computing module
        self.morphology_encoder = nn.Sequential(
            nn.Linear(config['field_dimension'], config['morphology']['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['morphology']['hidden_dim'], config['morphology']['state_dim'])
        )
        
        # Basic goal-directed behavior
        self.goal_network = nn.Sequential(
            nn.Linear(config['morphology']['state_dim'] + config['field_dimension'], 
                     config['goals']['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['goals']['hidden_dim'], config['goals']['num_goals'])
        )
        
        # Initialize state
        self.reset_state()
        
    def _create_ion_channel(self, config: Dict) -> nn.Module:
        """Create an ion channel network module"""
        return nn.Sequential(
            nn.Linear(config['input_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['output_dim']),
            nn.Sigmoid()  # Ion channels open/close with sigmoid activation
        )
    
    def reset_state(self):
        """Reset the bioelectric state to resting potential"""
        field_dim = self.config['field_dimension']
        self.state = BioelectricState(
            voltage_potential=torch.zeros(field_dim),
            ion_gradients={
                'sodium': torch.ones(field_dim) * 0.1,
                'potassium': torch.ones(field_dim) * 0.9,
                'calcium': torch.ones(field_dim) * 0.2
            },
            gap_junction_states=torch.eye(field_dim),
            morphological_state=torch.zeros(self.config['morphology']['state_dim'])
        )
        
    def process_stimulus(self, stimulus: torch.Tensor) -> Tuple[BioelectricState, Dict]:
        """Process external stimulus and update bioelectric state"""
        # Update ion channel states
        for ion, channel in self.ion_channels.items():
            channel_response = channel(stimulus)
            self.state.ion_gradients[ion] = channel_response
            
        # Calculate new membrane potential
        voltage_contribution = sum(
            grad * self.config['ion_weights'][ion] 
            for ion, grad in self.state.ion_gradients.items()
        )
        self.state.voltage_potential = voltage_contribution
        
        # Cell-to-cell communication via gap junctions
        field_communication = self.gap_junction_network(self.state.voltage_potential)
        
        # Update morphological state based on bioelectric pattern
        new_morphology = self.morphology_encoder(field_communication)
        self.state.morphological_state = 0.9 * self.state.morphological_state + 0.1 * new_morphology
        
        # Calculate goal-directed behavior
        combined_state = torch.cat([self.state.voltage_potential, self.state.morphological_state])
        goal_signals = self.goal_network(combined_state)
        
        # Homeostasis mechanism - try to return to stable state
        homeostatic_drive = torch.abs(self.state.voltage_potential - 
                                     torch.tensor(self.config['resting_potential'])).mean()
        
        return self.state, {
            'goal_signals': goal_signals,
            'homeostatic_drive': homeostatic_drive,
            'bioelectric_complexity': self._calculate_complexity()
        }
    
    def _calculate_complexity(self) -> float:
        """Calculate bioelectric complexity measure"""
        # Measure pattern complexity in the voltage field
        fft_result = torch.fft.rfft(self.state.voltage_potential)
        power_spectrum = torch.abs(fft_result)**2
        normalized_spectrum = power_spectrum / power_spectrum.sum()
        
        # Calculate spectral entropy (complexity measure)
        entropy = -torch.sum(normalized_spectrum * torch.log2(normalized_spectrum + 1e-10))
        
        return entropy.item()