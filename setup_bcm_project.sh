#!/bin/bash
# Script to create the bioelectric-consciousness-module project structure

# Create main project directory
mkdir -p bioelectric-consciousness-module
cd bioelectric-consciousness-module

# Create the basic files in the root directory
echo "# Bioelectric Consciousness Module (BCM)

## ⚠️ Important Disclaimer
This project is **NOT** affiliated with or endorsed by Dr. Michael Levin or his research group. The Bioelectric Consciousness Module (BCM) is inspired by scientific concepts from Dr. Levin's work but represents our own interpretation and implementation for research and learning purposes only.

## Overview
The Bioelectric Consciousness Module (BCM) is an experimental project exploring the computational implementation of primitive consciousness based on bioelectric principles observed in simple cellular organisms." > README.md

echo "MIT License

Copyright (c) $(date +%Y) [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files..." > LICENSE

echo "torch>=1.12.0
numpy>=1.20.0
matplotlib>=3.5.0
pyyaml>=6.0
tqdm>=4.62.0
scipy>=1.7.0
scikit-learn>=1.0.0
pytest>=7.0.0
jupyterlab>=3.0.0
networkx>=2.6.0
wandb>=0.12.0" > requirements.txt

echo "# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
dist/
build/
*.egg-info/

# Virtual environments
venv/
env/
ENV/

# IDE files
.idea/
.vscode/
*.swp
*.swo

# Output files
output/
*.png
*.jpg
*.mp4
*.log" > .gitignore

# Create directory structure
mkdir -p configs docs bcm/{core,morphology,collective,goals,memory,bridge,utils} \
        simulations/{environments,scenarios,evaluation} \
        experiments scripts tests

# Create __init__.py files in all Python package directories
find . -type d -not -path './.*' -not -path './docs*' -not -path './configs*' -not -path './scripts*' | while read dir; do
    touch "$dir/__init__.py"
done

# Create config files
cat > configs/bioelectric_config.yaml << 'EOF'
# Bioelectric Core Configuration

# Core bioelectric parameters
field_dimension: 64
bioelectric_channels: 8
signaling_layers: 3
gap_junction_heads: 4
gap_junction_dropout: 0.1
resting_potential: 0.2
activation_threshold: 0.6

# Ion channel parameters
sodium_channel:
  input_dim: 32
  hidden_dim: 64
  output_dim: 64
  conductance: 0.8

potassium_channel:
  input_dim: 32
  hidden_dim: 64
  output_dim: 64
  conductance: 0.6

calcium_channel:
  input_dim: 32
  hidden_dim: 64
  output_dim: 64
  conductance: 0.3

# Ion weights for potential calculation
ion_weights:
  sodium: 1.0
  potassium: -0.8
  calcium: 0.5

# Morphology parameters
morphology:
  state_dim: 32
  hidden_dim: 64
  adaptation_rate: 0.01
  plasticity: 0.3
  stability_preference: 0.7

# Goal-directed behavior parameters
goals:
  num_goals: 4  # Basic cellular drives: nutrients, avoid harm, homeostasis, reproduction
  hidden_dim: 32
  survival_priority: 0.8
  
# Bridging parameters for ACM integration
emotion:
  base_dimension: 16
  valence_dim: 8
  arousal_dim: 4
  dominance_dim: 4
EOF

cat > configs/simulation_config.yaml << 'EOF'
# Simulation environment configuration

environment:
  size: [64, 64]  # 2D grid size
  update_rate: 0.05
  
  # Nutrient field parameters
  nutrients:
    initial_concentration: 0.5
    diffusion_rate: 0.02
    regeneration_rate: 0.01
    
  # Perturbation parameters
  perturbation:
    probability: 0.05
    strength_mean: 0.3
    strength_std: 0.1
    
  # Physical constraints
  physics:
    viscosity: 0.7
    temperature: 0.5  # Normalized temperature (0-1)
EOF

cat > configs/bridge_config.yaml << 'EOF'
# Configuration for BCM to ACM bridge

# Bridge parameters
field_dimension: 64
emotion:
  base_dimension: 16
  valence_dim: 8
  arousal_dim: 4
  dominance_dim: 4

# Export parameters
export:
  format_version: "0.1.0"
  compress_output: false
  include_raw_states: true
  
# ACM compatibility mappings
acm_mappings:
  emotional_context:
    valence: "valence"
    arousal: "arousal"
    stability: "coherence" 
    dominance: "homeostatic_balance"
EOF

# Create core bioelectric module
cat > bcm/core/bioelectric_core.py << 'EOF'
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
EOF

# Create the bridge module
cat > bcm/bridge/bcm_to_acm.py << 'EOF'
"""
Bridge module that translates BCM bioelectric states to ACM-compatible primitives.
This enables the integration of primitive bioelectric consciousness with more
advanced emotional and cognitive architectures.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

from bcm.core.bioelectric_core import BioelectricState

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
                'pattern_stability': self._calculate_stability(bioelectric_state),
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
EOF

# Create a basic environment module
cat > simulations/environments/nutrient_field.py << 'EOF'
"""
Implements a nutrient field environment for simulating cell-like entities.
Provides external stimuli for the bioelectric consciousness module.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Any

class NutrientEnvironment:
    """
    Simulates a 2D field with nutrients and perturbations to provide
    stimuli for bioelectric systems.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.size = config.get('size', [64, 64])
        
        # Initialize nutrient field
        self.nutrient_field = torch.ones(self.size) * config['nutrients']['initial_concentration']
        
        # Parameters
        self.diffusion_rate = config['nutrients']['diffusion_rate']
        self.regeneration_rate = config['nutrients']['regeneration_rate']
        self.perturbation_prob = config['perturbation']['probability']
        self.perturbation_mean = config['perturbation']['strength_mean']
        self.perturbation_std = config['perturbation']['strength_std']
        
        # State tracking
        self.step_count = 0
        
    def get_stimulus(self, step: int = None) -> torch.Tensor:
        """
        Generate stimulus vector from the current state of the environment.
        This will be processed by the bioelectric system.
        """
        if step is not None:
            self.step_count = step
            
        # Flatten nutrient field to a vector stimulus
        stimulus = self.nutrient_field.reshape(-1)
        
        # Add random perturbation with configured probability
        if np.random.random() < self.perturbation_prob:
            perturbation = torch.normal(
                mean=self.perturbation_mean,
                std=self.perturbation_std,
                size=stimulus.shape
            )
            stimulus = stimulus + perturbation
        
        # Ensure stimulus is in valid range
        stimulus = torch.clamp(stimulus, 0.0, 1.0)
        
        return stimulus
    
    def update(self, response: torch.Tensor) -> None:
        """
        Update environment based on system response.
        Simulates consumption of nutrients and diffusion.
        """
        # Reshape response to match field if needed
        if response.shape != self.nutrient_field.shape:
            response = response[:self.size[0] * self.size[1]].reshape(self.size)
        
        # Simulate consumption of nutrients based on response
        consumption = response * 0.1  # Scale response to consumption rate
        self.nutrient_field = torch.clamp(self.nutrient_field - consumption, 0.0, 1.0)
        
        # Simulate diffusion
        self._diffuse()
        
        # Regenerate nutrients
        self._regenerate()
        
        self.step_count += 1
    
    def _diffuse(self) -> None:
        """Simulate diffusion of nutrients across the field"""
        # Simple diffusion using convolution
        kernel = torch.tensor([
            [0.05, 0.2, 0.05],
            [0.2, 0.0, 0.2],
            [0.05, 0.2, 0.05]
        ])
        
        # Unimplemented placeholder - in a real implementation 
        # you would use torch.nn.functional.conv2d for proper diffusion
        # For simplicity, we'll just add a small random factor
        diffusion = torch.randn_like(self.nutrient_field) * self.diffusion_rate
        self.nutrient_field = torch.clamp(self.nutrient_field + diffusion, 0.0, 1.0)
    
    def _regenerate(self) -> None:
        """Regenerate nutrients at a slow rate"""
        regeneration = torch.ones_like(self.nutrient_field) * self.regeneration_rate
        self.nutrient_field = torch.clamp(self.nutrient_field + regeneration, 0.0, 1.0)
EOF

# Create a utility module for metrics
cat > bcm/utils/metrics.py << 'EOF'
"""
Utility functions for calculating metrics on bioelectric states.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from bcm.core.bioelectric_core import BioelectricState

def calculate_pattern_stability(bioelectric_state: BioelectricState) -> float:
    """
    Calculate stability of the bioelectric pattern.
    Higher values indicate more stable patterns.
    """
    voltage_pattern = bioelectric_state.voltage_potential
    
    # Calculate spatial stability (local consistency)
    if len(voltage_pattern) <= 1:
        return 1.0  # Single element is perfectly stable
        
    diffs = torch.abs(voltage_pattern[1:] - voltage_pattern[:-1])
    spatial_stability = 1.0 / (1.0 + diffs.mean().item())
    
    return spatial_stability

def calculate_pattern_complexity(bioelectric_state: BioelectricState) -> float:
    """
    Calculate complexity of the bioelectric pattern.
    Higher values indicate more complex patterns.
    """
    voltage_pattern = bioelectric_state.voltage_potential
    
    # Use FFT to measure frequency components
    fft_result = torch.fft.rfft(voltage_pattern)
    power_spectrum = torch.abs(fft_result)**2
    
    # Normalize the spectrum
    if power_spectrum.sum() > 0:
        normalized_spectrum = power_spectrum / power_spectrum.sum()
    else:
        return 0.0
    
    # Calculate spectral entropy as complexity measure
    entropy = -torch.sum(normalized_spectrum * torch.log2(normalized_spectrum + 1e-10))
    
    # Normalize to 0-1 range (assuming max entropy for a signal of this length)
    max_entropy = np.log2(len(normalized_spectrum))
    normalized_entropy = entropy.item() / max_entropy if max_entropy > 0 else 0
    
    return normalized_entropy

def calculate_responsiveness(
    stimulus: torch.Tensor, 
    response_pattern: torch.Tensor
) -> float:
    """
    Calculate responsiveness of the system to stimuli.
    Higher values indicate stronger stimulus-response relationship.
    """
    # Simple correlation-based responsiveness
    # Normalize vectors
    if len(stimulus) != len(response_pattern):
        # If dimensions don't match, use smaller length
        min_length = min(len(stimulus), len(response_pattern))
        stimulus = stimulus[:min_length]
        response_pattern = response_pattern[:min_length]
    
    # Calculate correlation
    stim_norm = stimulus - stimulus.mean()
    resp_norm = response_pattern - response_pattern.mean()
    
    stim_std = torch.std(stim_norm)
    resp_std = torch.std(resp_norm)
    
    if stim_std > 0 and resp_std > 0:
        correlation = torch.mean(stim_norm * resp_norm) / (stim_std * resp_std)
        # Convert to 0-1 range
        responsiveness = (correlation + 1) / 2
    else:
        responsiveness = 0.5  # Default if no variation
    
    return responsiveness.item()
EOF

# Create a visualization module
cat > bcm/utils/visualization.py << 'EOF'
"""
Visualization utilities for bioelectric states and metrics.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
from bcm.core.bioelectric_core import BioelectricState

def save_state_visualization(
    bioelectric_state: BioelectricState, 
    metrics: Dict, 