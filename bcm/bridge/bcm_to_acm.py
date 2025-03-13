"""
Bridge module that translates BCM bioelectric states to ACM-compatible primitives.
This enables the integration of primitive bioelectric consciousness with more
advanced emotional and cognitive architectures.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional

from bcm.core.bioelectric_core import BioelectricState
from bcm.utils.metrics import calculate_pattern_stability

class BioelectricToACMBridge:
    """
    Bridges primitive bioelectric consciousness states to ACM-compatible formats.
    
    This component translates bioelectric states into foundational
    emotional primitives that can be consumed by ACM.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Mapping between bioelectric patterns and primitive emotions
        self.pattern_emotion_map = nn.Sequential(
            nn.Linear(config['field_dimension'], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, config['emotion']['base_dimension'])
        )
        
    def translate_bioelectric_to_emotional(
        self, 
        bioelectric_state: BioelectricState
    ) -> Dict[str, Any]:
        """
        Translate bioelectric patterns to primitive emotional states
        that can be processed by the ACM emotion network.
        """
        # Extract key bioelectric patterns
        voltage_pattern = bioelectric_state.voltage_potential
        
        # Calculate primitive emotional response from bioelectric patterns
        primitive_emotions = self.pattern_emotion_map(voltage_pattern)
        
        # Map to basic emotional dimensions: 
        # valence (positive/negative), arousal (intensity), dominance
        valence = torch.tanh(primitive_emotions[:self.config['emotion']['valence_dim']].mean())
        arousal = torch.sigmoid(primitive_emotions[self.config['emotion']['valence_dim']:
                                                 self.config['emotion']['valence_dim'] + 
                                                 self.config['emotion']['arousal_dim']].mean())
        
        # Create emotion dictionary using ACM's expected format
        emotion_dict = {
            'valence': valence.item(),
            'arousal': arousal.item(),
            'stability': self._calculate_stability(bioelectric_state),
            'coherence': self._calculate_coherence(bioelectric_state)
        }
        
        # Create ACM-compatible output format
        acm_compatible = {
            'emotional_context': emotion_dict,
            'bioelectric_metadata': {
                'complexity': self._calculate_complexity(bioelectric_state),
                'pattern_stability': calculate_pattern_stability(bioelectric_state),
                'goal_orientation': self._calculate_goal_orientation(bioelectric_state)
            },
            'primitive_narrative': self._generate_primitive_narrative(bioelectric_state),
            'attention_focus': self._calculate_attention_signal(bioelectric_state)
        }
        
        return acm_compatible
    
    def _calculate_stability(self, bioelectric_state: BioelectricState) -> float:
        """Calculate stability as homeostatic balance"""
        return 1.0 - torch.abs(
            bioelectric_state.voltage_potential - 
            torch.tensor(self.config['resting_potential'])
        ).mean().item()
    
    def _calculate_coherence(self, bioelectric_state: BioelectricState) -> float:
        """Calculate coherence as pattern consistency across the field"""
        # Spatial coherence across the bioelectric field
        field_std = bioelectric_state.voltage_potential.std().item()
        coherence = 1.0 / (1.0 + field_std)
        return coherence
    
    def _calculate_complexity(self, bioelectric_state: BioelectricState) -> float:
        """Calculate bioelectric pattern complexity"""
        # Use FFT to measure pattern complexity
        fft_result = torch.fft.rfft(bioelectric_state.voltage_potential)
        power_spectrum = torch.abs(fft_result)**2
        normalized_spectrum = power_spectrum / power_spectrum.sum()
        
        # Calculate spectral entropy
        entropy = -torch.sum(normalized_spectrum * torch.log2(normalized_spectrum + 1e-10))
        return entropy.item()
    
    def _generate_primitive_narrative(self, bioelectric_state: BioelectricState) -> str:
        """Generate a simple narrative description of the bioelectric state"""
        voltage_mean = bioelectric_state.voltage_potential.mean().item()
        voltage_std = bioelectric_state.voltage_potential.std().item()
        
        if voltage_mean > 0.7:
            return "heightened activity state, seeking stimulus"
        elif voltage_mean > 0.4:
            return "moderate activity state, responsive to environment"
        elif voltage_mean > 0.2:
            return "baseline activity state, maintaining homeostasis"
        else:
            return "low activity state, conserving resources"
    
    def _calculate_goal_orientation(self, bioelectric_state: BioelectricState) -> float:
        """Calculate how strongly the system is oriented toward a goal"""
        # Simple implementation - can be expanded
        voltage_gradient = torch.diff(bioelectric_state.voltage_potential)
        return torch.abs(voltage_gradient).mean().item()
    
    def _calculate_attention_signal(self, bioelectric_state: BioelectricState) -> Dict:
        """Calculate primitive attention focus from bioelectric pattern"""
        # Find the region of highest voltage activity (simple attention mechanism)
        max_idx = torch.argmax(bioelectric_state.voltage_potential)
        attention_strength = bioelectric_state.voltage_potential[max_idx].item()
        
        return {
            'focus_region': max_idx.item(),
            'strength': attention_strength,
            'distribution': torch.softmax(bioelectric_state.voltage_potential, dim=0).tolist()
        }
        
    def export_acm_training_data(self, states_history: List[Dict]) -> Dict:
        """
        Export processed bioelectric states in a format suitable for ACM training.
        This enables the evolution from simple bioelectric patterns to complex emotions.
        """
        # Process state history into ACM training format
        acm_training_data = []
        
        for state_record in states_history:
            bioelectric_state = state_record['bioelectric_state']
            stimulus = state_record['stimulus']
            response = state_record['response']
            
            # Translate to ACM format
            acm_entry = self.translate_bioelectric_to_emotional(bioelectric_state)
            acm_entry.update({
                'stimulus_embedding': stimulus.tolist() if hasattr(stimulus, 'tolist') else stimulus,
                'response_pattern': response.tolist() if hasattr(response, 'tolist') else response,
                'timestamp': state_record.get('timestamp', 0)
            })
            
            acm_training_data.append(acm_entry)
        
        return {
            'acm_training_data': acm_training_data,
            'metadata': {
                'version': '0.1.0',
                'source': 'BCM',
                'records': len(acm_training_data)
            }
        }