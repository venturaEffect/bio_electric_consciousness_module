"""
Implementation of collective cell colony behavior.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np

from bcm.core.bioelectric_core import BioelectricConsciousnessCore, BioelectricState

class CellColony(nn.Module):
    """
    Models a colony of cells with collective bioelectric behavior.
    
    Implements collective intelligence principles where individual
    cells communicate through bioelectric signaling to produce
    coordinated responses that no single cell could achieve alone.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.num_cells = config.get('num_cells', 8)
        
        # Create individual cell instances
        self.cells = nn.ModuleList([
            BioelectricConsciousnessCore(config) 
            for _ in range(self.num_cells)
        ])
        
        # Intercellular communication network
        self.communication_network = nn.Parameter(
            torch.ones(self.num_cells, self.num_cells) * 0.1
        )
        
        # Colony integration layer
        field_dim = config['field_dimension']
        self.colony_integration = nn.Sequential(
            nn.Linear(field_dim * self.num_cells, field_dim * 2),
            nn.ReLU(),
            nn.Linear(field_dim * 2, field_dim)
        )
        
    def process_colony_stimulus(self, stimulus: torch.Tensor) -> Tuple[List[BioelectricState], Dict]:
        """
        Process stimulus through cell colony
        
        Args:
            stimulus: Environmental stimulus
            
        Returns:
            Tuple of (cell_states, colony_metrics)
        """
        # Process stimulus through each cell
        cell_states = []
        cell_outputs = []
        
        for i, cell in enumerate(self.cells):
            # Each cell might receive slightly different stimulus based on its position
            cell_stimulus = stimulus + torch.randn_like(stimulus) * 0.01
            
            # Process through cell
            state, outputs = cell.process_stimulus(cell_stimulus)
            
            cell_states.append(state)
            cell_outputs.append(outputs)
        
        # Facilitate intercellular communication
        communication_matrix = torch.sigmoid(self.communication_network)
        
        for i in range(self.num_cells):
            # Collect signals from other cells
            incoming_signals = torch.zeros_like(cell_states[0].voltage_potential)
            
            for j in range(self.num_cells):
                if i != j:
                    signal_strength = communication_matrix[i, j]
                    incoming_signals += signal_strength * cell_states[j].voltage_potential
            
            # Update cell state with signals from other cells
            cell_states[i].voltage_potential = (
                0.8 * cell_states[i].voltage_potential +
                0.2 * incoming_signals
            )
        
        # Integrate colony-level metrics
        colony_voltage = torch.cat([state.voltage_potential for state in cell_states])
        colony_representation = self.colony_integration(colony_voltage)
        
        # Calculate colony-level metrics
        avg_complexity = sum(output['bioelectric_complexity'] for output in cell_outputs) / self.num_cells
        voltage_coherence = self._calculate_coherence([state.voltage_potential for state in cell_states])
        
        colony_metrics = {
            'colony_representation': colony_representation,
            'avg_complexity': avg_complexity,
            'voltage_coherence': voltage_coherence,
            'communication_density': communication_matrix.mean().item()
        }
        
        return cell_states, colony_metrics
    
    def _calculate_coherence(self, voltage_patterns: List[torch.Tensor]) -> float:
        """
        Calculate coherence between voltage patterns across cells
        
        Args:
            voltage_patterns: List of voltage patterns from each cell
            
        Returns:
            Coherence measure between 0 and 1
        """
        if not voltage_patterns:
            return 0.0
            
        # Calculate pairwise correlations
        correlations = []
        
        for i in range(len(voltage_patterns)):
            for j in range(i+1, len(voltage_patterns)):
                v1 = voltage_patterns[i]
                v2 = voltage_patterns[j]
                
                # Calculate correlation
                v1_norm = v1 - v1.mean()
                v2_norm = v2 - v2.mean()
                
                v1_std = torch.sqrt(torch.sum(v1_norm**2))
                v2_std = torch.sqrt(torch.sum(v2_norm**2))
                
                if v1_std > 0 and v2_std > 0:
                    corr = torch.sum(v1_norm * v2_norm) / (v1_std * v2_std)
                    correlations.append(corr.item())
                else:
                    correlations.append(0.0)
        
        # Average correlation as coherence measure
        if correlations:
            return sum(correlations) / len(correlations)
        else:
            return 0.0

    def forward(self, state: BioelectricState) -> BioelectricState:
        """
        Process a bioelectric state through the cell colony.
        
        Args:
            state: Current bioelectric state
            
        Returns:
            Updated bioelectric state after colony interactions
        """
        # Extract a stimulus representation from the current state
        stimulus = state.voltage_potential.flatten()
        
        # Process through colony
        cell_states, _ = self.process_colony_stimulus(stimulus)
        
        # Aggregate colony behavior into unified state
        # Use the first cell as a base and incorporate colony-wide effects
        updated_state = BioelectricState(
            voltage_potential=cell_states[0].voltage_potential.clone(),
            ion_gradients={
                ion: grad.clone() 
                for ion, grad in cell_states[0].ion_gradients.items()
            },
            gap_junction_states=state.gap_junction_states.clone(),
            morphological_state=state.morphological_state.clone()
        )
        
        # Apply colony-wide effects (average voltage potentials across cells)
        avg_voltage = torch.stack([cs.voltage_potential for cs in cell_states]).mean(dim=0)
        updated_state.voltage_potential = avg_voltage
        
        return updated_state