from flask import Flask, render_template, request, jsonify
import os
import sys
import torch
import numpy as np
import json
import yaml
import logging
from typing import Dict, Any, List, Optional
import copy
import datetime

# Add parent directory to path so we can import BCM modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import BCM modules
from bcm.core.bioelectric_core import BioelectricConsciousnessCore, BioelectricState
from bcm.collective.cell_colony import CellColony
from bcm.morphology.pattern_formation import BioelectricPatternFormation
from bcm.goals.homeostasis import HomeostasisRegulator
import bcm.utils.visualization as viz

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('bcm_web')

app = Flask(__name__)

# Define helper functions first
def create_default_config():
    """
    Create a complete default configuration with all required keys.
    
    Returns:
        dict: Complete configuration with sensible defaults
    """
    config = {
        'core': {
            'field_dimension': 10,
            'membrane_capacitance': 0.5,
            'resting_potential': -0.2,
            'threshold': 0.3,
            'sodium_channel': {
                'conductance': 0.8,
                'reversal_potential': 1.0,
                'activation_threshold': 0.2,
                'inactivation_rate': 0.1,
                'layers': [64, 32, 1]
            },
            'potassium_channel': {
                'conductance': 0.6,
                'reversal_potential': -0.8,
                'activation_threshold': 0.1,
                'inactivation_rate': 0.05,
                'layers': [64, 32, 1]
            },
            'calcium_channel': {
                'conductance': 0.4,
                'reversal_potential': 0.9,
                'activation_threshold': 0.3,
                'inactivation_rate': 0.15,
                'layers': [64, 32, 1]
            },
            'gap_junction_strength': 0.5
        },
        'pattern_formation': {
            'field_dimension': 10,
            'pattern_complexity': 0.7,
            'reaction_rate': 0.3,
            'diffusion_rate': 0.1,
            'boundary_condition': 'circular',
            # Add the missing morphology section
            'morphology': {
                'stability_preference': 0.7,
                'growth_rate': 0.2,
                'polarization_threshold': 0.3,
                'pattern_scale': 1.0
            }
        },
        'colony': {
            'num_cells': 5,
            'connection_density': 0.8,
            'communication_strength': 0.6,
            'response_threshold': 0.2,
            'field_dimension': 10
        },
        'homeostasis': {
            'homeostasis_strength': 0.5,
            'resting_potential': -0.2,
            'target_ion_levels': {
                'sodium': 0.1,
                'potassium': 0.8,
                'calcium': 0.2
            },
            'correction_rate': 0.1,
            'field_dimension': 10
        }
    }
    return config

def update_config(base_config, updates):
    """
    Updates configuration with provided parameter changes.
    
    Args:
        base_config (dict): Base configuration dictionary
        updates (dict): Updates to apply
        
    Returns:
        dict: Updated configuration
    """
    # Make a deep copy to avoid modifying the original
    config = copy.deepcopy(base_config)
    
    # Apply updates
    for section, params in updates.items():
        if section not in config:
            config[section] = {}
        for name, value in params.items():
            config[section][name] = value
    
    return config

# Now load default configuration
default_config = create_default_config()

# Try to load from file if available, but fall back to defaults
try:
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'configs', 'bioelectric_config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
            # Merge file config with defaults to ensure all required keys exist
            default_config = update_config(default_config, file_config)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        logger.warning(f"Config file not found at {config_path}, using defaults")
except Exception as e:
    logger.warning(f"Error loading config from file: {str(e)}. Using defaults.")

# Global model instances and device
models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Global state
current_state = None

# Define the simplified bioelectric core model for web interface
class SimplifiedBioelectricCore(torch.nn.Module):
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

class SimplifiedPatternFormation(torch.nn.Module):
    """
    A simplified version of the BioelectricPatternFormation for web visualization.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pattern_complexity = config.get('pattern_complexity', 0.7)
        self.reaction_rate = config.get('reaction_rate', 0.3)
        self.diffusion_rate = config.get('diffusion_rate', 0.1)
        self.field_dimension = config.get('field_dimension', 10)
        
    def forward(self, state):
        """Process a state through simplified pattern formation dynamics."""
        # Extract morphological state and voltage potential
        morphological_state = state.morphological_state
        voltage = state.voltage_potential
        
        # Convert to appropriate format if needed
        if isinstance(morphological_state, torch.Tensor):
            morph_flat = morphological_state.flatten()
        else:
            morph_flat = torch.tensor(morphological_state, device=voltage.device).flatten()
        
        # Resize if needed
        field_dim = self.field_dimension
        if len(morph_flat) < field_dim*field_dim:
            # Pad with zeros
            padding = torch.zeros(field_dim*field_dim - len(morph_flat), device=morph_flat.device)
            morph_flat = torch.cat([morph_flat, padding])
        elif len(morph_flat) > field_dim*field_dim:
            # Truncate
            morph_flat = morph_flat[:field_dim*field_dim]
        
        # Reshape to 2D
        morph_2d = morph_flat.reshape(field_dim, field_dim)
        
        # Apply simple pattern formation rule (simplified reaction-diffusion)
        voltage_influence = 0.1 * (voltage - voltage.mean())
        morph_laplacian = self._compute_laplacian(morph_2d)
        
        # Update morphological state
        updated_morph = morph_2d + self.reaction_rate * voltage_influence + self.diffusion_rate * morph_laplacian
        updated_morph = torch.tanh(updated_morph)  # Bound values
        
        # Create updated state
        new_state = BioelectricState(
            voltage_potential=state.voltage_potential,
            ion_gradients=state.ion_gradients,
            gap_junction_states=state.gap_junction_states,
            morphological_state=updated_morph.flatten()
        )
        
        return new_state
    
    def _compute_laplacian(self, tensor):
        """Compute discrete Laplacian operator (for diffusion)."""
        # Get dimensions
        h, w = tensor.shape
        
        # Compute discrete Laplacian
        laplacian = torch.zeros_like(tensor)
        
        # Internal points
        for i in range(1, h-1):
            for j in range(1, w-1):
                laplacian[i, j] = (
                    tensor[i+1, j] + tensor[i-1, j] + 
                    tensor[i, j+1] + tensor[i, j-1] - 
                    4 * tensor[i, j]
                )
                
        # Boundary conditions (no-flux)
        for j in range(1, w-1):
            laplacian[0, j] = 2 * tensor[1, j] + tensor[0, j+1] + tensor[0, j-1] - 4 * tensor[0, j]
            laplacian[h-1, j] = 2 * tensor[h-2, j] + tensor[h-1, j+1] + tensor[h-1, j-1] - 4 * tensor[h-1, j]
        
        for i in range(1, h-1):
            laplacian[i, 0] = tensor[i+1, 0] + tensor[i-1, 0] + 2 * tensor[i, 1] - 4 * tensor[i, 0]
            laplacian[i, w-1] = tensor[i+1, w-1] + tensor[i-1, w-1] + 2 * tensor[i, w-2] - 4 * tensor[i, w-1]
        
        # Corners
        laplacian[0, 0] = 2 * tensor[1, 0] + 2 * tensor[0, 1] - 4 * tensor[0, 0]
        laplacian[0, w-1] = 2 * tensor[1, w-1] + 2 * tensor[0, w-2] - 4 * tensor[0, w-1]
        laplacian[h-1, 0] = 2 * tensor[h-2, 0] + 2 * tensor[h-1, 1] - 4 * tensor[h-1, 0]
        laplacian[h-1, w-1] = 2 * tensor[h-2, w-1] + 2 * tensor[h-1, w-2] - 4 * tensor[h-1, w-1]
        
        return laplacian

class SimplifiedColony(torch.nn.Module):
    """
    A simplified version of the CellColony for web visualization.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.communication_strength = config.get('communication_strength', 0.6)
        
    def forward(self, state):
        """Process a state through simplified colony interactions."""
        # For simplicity, just apply a small amount of global communication
        new_voltage = state.voltage_potential * (1.0 - self.communication_strength) + \
                     state.voltage_potential.mean() * self.communication_strength
        
        # Create updated state
        new_state = BioelectricState(
            voltage_potential=new_voltage,
            ion_gradients=state.ion_gradients,
            gap_junction_states=state.gap_junction_states,
            morphological_state=state.morphological_state
        )
        
        return new_state

class SimplifiedHomeostasis(torch.nn.Module):
    """
    A simplified version of the HomeostasisRegulator for web visualization.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.homeostasis_strength = config.get('homeostasis_strength', 0.5)
        self.resting_potential = config.get('resting_potential', -0.2)
        self.target_ion_levels = config.get('target_ion_levels', {
            'sodium': 0.1,
            'potassium': 0.8,
            'calcium': 0.2
        })
        
    def forward(self, state):
        """Apply simple homeostatic correction."""
        # Pull voltage toward resting potential
        voltage_error = state.voltage_potential - self.resting_potential
        correction = -self.homeostasis_strength * voltage_error
        new_voltage = state.voltage_potential + correction
        
        # Apply simple homeostasis to ion gradients
        new_ion_gradients = {}
        for ion, grad in state.ion_gradients.items():
            if ion in self.target_ion_levels:
                target = self.target_ion_levels[ion]
                error = grad - target
                ion_correction = -self.homeostasis_strength * 0.5 * error
                new_ion_gradients[ion] = grad + ion_correction
            else:
                new_ion_gradients[ion] = grad
        
        # Create updated state
        new_state = BioelectricState(
            voltage_potential=new_voltage,
            ion_gradients=new_ion_gradients,
            gap_junction_states=state.gap_junction_states,
            morphological_state=state.morphological_state
        )
        
        return new_state

def init_models(config):
    """Initialize models based on configuration."""
    global models
    
    logger.info("Initializing models with current configuration")
    
    # Initialize core model
    try:
        models['core'] = BioelectricConsciousnessCore(config['core']).to(device)
    except Exception as e:
        logger.warning(f"Failed to initialize full core model: {str(e)}. Using simplified version.")
        models['core'] = SimplifiedBioelectricCore(config['core']).to(device)
    
    # Initialize pattern model
    try:
        models['pattern_model'] = BioelectricPatternFormation(config['pattern_formation']).to(device)
    except Exception as e:
        logger.warning(f"Failed to initialize pattern formation model: {str(e)}. Using simplified version.")
        models['pattern_model'] = SimplifiedPatternFormation(config['pattern_formation']).to(device)
    
    # Initialize colony model
    try:
        models['colony'] = CellColony(config['colony']).to(device)
    except Exception as e:
        logger.warning(f"Failed to initialize colony model: {str(e)}. Using simplified version.")
        models['colony'] = SimplifiedColony(config['colony']).to(device)
    
    # Initialize homeostasis model
    try:
        models['homeostasis'] = HomeostasisRegulator(config['homeostasis']).to(device)
    except Exception as e:
        logger.warning(f"Failed to initialize homeostasis model: {str(e)}. Using simplified version.")
        models['homeostasis'] = SimplifiedHomeostasis(config['homeostasis']).to(device)
    
    return models

# Initialize with default config
init_models(default_config)

@app.route('/')
def index():
    """Main page with simulation interface."""
    return render_template('index.html')

@app.route('/api/init_simulation', methods=['POST'])
def init_simulation():
    """Initialize the simulation with chosen parameters."""
    data = request.json
    scenario = data.get('scenario', 'default')
    
    # Get grid size from config
    grid_size = default_config['core'].get('field_dimension', 10)
    
    # Initialize with small random values
    voltage_potential = torch.randn(grid_size, grid_size, device=device) * 0.01
    
    # Apply scenario-specific initializations
    if scenario == 'two_heads':
        # Add two high-voltage regions suggesting two head formation sites
        voltage_potential[1:3, 1:3] = 0.8  # Top head voltage pattern
        voltage_potential[7:9, 7:9] = 0.8  # Bottom head voltage pattern
    elif scenario == 'eye_formation':
        # Setup pattern for ectopic eye formation
        voltage_potential[4:6, 4:6] = 0.9  # Central eye-inducing voltage pattern
    
    ion_gradients = {
        'sodium': torch.randn(grid_size, grid_size, device=device) * 0.01,
        'potassium': torch.randn(grid_size, grid_size, device=device) * 0.01,
        'calcium': torch.randn(grid_size, grid_size, device=device) * 0.01,
    }
    
    morphological_state = torch.zeros(grid_size*grid_size, device=device)
    
    # Create state dictionary for JSON serialization
    global current_state
    current_state = {
        'voltage_potential': voltage_potential.cpu().numpy().tolist(),
        'ion_gradients': {
            ion: grad.cpu().numpy().tolist() for ion, grad in ion_gradients.items()
        },
        'morphological_state': morphological_state.cpu().numpy().tolist()
    }
    
    return jsonify({'success': True, 'state': current_state})

@app.route('/api/run_step', methods=['POST'])
def run_step():
    """Run a single simulation step with provided parameters."""
    try:
        logger.info("Received run_step request")
        data = request.json
        config_updates = data.get('config_updates', {})
        state_data = data.get('state', None)
        scenario = data.get('scenario', 'default')
        
        logger.info(f"Processing scenario: {scenario}, with state: {state_data is not None}")
        
        # Update config with provided parameters
        updated_config = update_config(default_config, config_updates)
        
        # Re-initialize models if config changed significantly
        if config_updates:
            init_models(updated_config)
        
        # Create initial state if not provided
        if not state_data:
            # Create default initial state
            grid_size = updated_config.get('core', {}).get('field_dimension', 10)
            
            # Initialize with small random values
            voltage_potential = torch.randn(grid_size, grid_size, device=device) * 0.01
            
            # Add a specific pattern for demonstration
            if scenario == 'two_heads':
                # Add two high-voltage regions suggesting two head formation sites
                voltage_potential[1:3, 1:3] = 0.8  # Top head voltage pattern
                voltage_potential[7:9, 7:9] = 0.8  # Bottom head voltage pattern
            elif scenario == 'regeneration':
                # Add gradient pattern simulating a cut area
                for i in range(grid_size):
                    for j in range(grid_size):
                        if i > grid_size // 2:
                            voltage_potential[i, j] = 0.5 * (1 - (i - grid_size//2) / (grid_size//2))
            
            ion_gradients = {
                'sodium': torch.randn(grid_size, grid_size, device=device) * 0.01,
                'potassium': torch.randn(grid_size, grid_size, device=device) * 0.01,
                'calcium': torch.randn(grid_size, grid_size, device=device) * 0.01,
            }
            
            morphological_state = torch.zeros(grid_size*grid_size, device=device)
            
            # Create state dictionary for JSON serialization
            state_data = {
                'voltage_potential': voltage_potential.cpu().numpy().tolist(),
                'ion_gradients': {
                    ion: grad.cpu().numpy().tolist() for ion, grad in ion_gradients.items()
                },
                'morphological_state': morphological_state.cpu().numpy().tolist()
            }
            logger.info(f"Created initial state with dimensions: {len(state_data['voltage_potential'])}x{len(state_data['voltage_potential'][0])}")
        
        # Convert state back to tensors for processing
        voltage_potential = torch.tensor(state_data['voltage_potential'], device=device)
        ion_gradients = {
            ion: torch.tensor(grad, device=device) 
            for ion, grad in state_data['ion_gradients'].items()
        }
        morphological_state = torch.tensor(state_data['morphological_state'], device=device)
        
        # Process state through models
        try:
            # Create BioelectricState object
            state = BioelectricState(
                voltage_potential=voltage_potential,
                ion_gradients=ion_gradients,
                gap_junction_states=torch.ones_like(voltage_potential),
                morphological_state=morphological_state
            )
            
            # Process through models
            if 'core' in models:
                state = models['core'](state)
            if 'pattern_model' in models:
                state = models['pattern_model'](state)
            if 'colony' in models:
                state = models['colony'](state)
            if 'homeostasis' in models:
                state = models['homeostasis'](state)
            
            # Convert back to Python native types for JSON serialization
            result_state = {
                'voltage_potential': state.voltage_potential.cpu().numpy().tolist(),
                'ion_gradients': {
                    ion: grad.cpu().numpy().tolist() for ion, grad in state.ion_gradients.items()
                },
                'morphological_state': state.morphological_state.cpu().numpy().tolist()
            }
            
            logger.info(f"Processed state, voltage shape: {len(result_state['voltage_potential'])}x{len(result_state['voltage_potential'][0])}")
            
            return jsonify({
                'success': True,
                'state': result_state
            })
            
        except Exception as e:
            logger.error(f"Error in model processing: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            })
            
    except Exception as e:
        logger.error(f"Error in run_step: {str(e)}")
        return jsonify({
            'success': False, 
            'error': str(e)
        })

@app.route('/api/scenarios', methods=['GET'])
def get_scenarios():
    """Return available experimental scenarios."""
    scenarios = [
        {
            'id': 'default',
            'name': 'Default Pattern',
            'description': 'Standard bioelectric pattern formation'
        },
        {
            'id': 'two_heads',
            'name': 'Two-Headed Planarian',
            'description': 'Simulate Levin\'s experiment creating two-headed planarians'
        },
        {
            'id': 'eye_formation',
            'name': 'Ectopic Eye Formation',
            'description': 'Simulate eye formation in non-head tissue'
        }
    ]
    return jsonify(scenarios)

@app.route('/api/parameters', methods=['GET'])
def get_parameters():
    """Return configurable parameters."""
    parameters = [
        {
            'section': 'core',
            'name': 'gap_junction_strength',
            'label': 'Gap Junction Strength',
            'description': 'Controls cell-to-cell electrical coupling',
            'min': 0.0,
            'max': 1.0,
            'step': 0.1,
            'default': default_config['core'].get('gap_junction_strength', 0.5)
        },
        {
            'section': 'pattern_formation',
            'name': 'pattern_complexity',
            'label': 'Pattern Complexity',
            'description': 'Controls complexity of emergent patterns',
            'min': 0.1,
            'max': 1.0,
            'step': 0.1,
            'default': default_config['pattern_formation'].get('pattern_complexity', 0.7)
        },
        {
            'section': 'homeostasis',
            'name': 'homeostasis_strength',
            'label': 'Homeostasis Strength',
            'description': 'How strongly system maintains target states',
            'min': 0.0,
            'max': 1.0,
            'step': 0.1,
            'default': default_config['homeostasis'].get('homeostasis_strength', 0.5)
        }
    ]
    return jsonify(parameters)

@app.route('/api/modify_pattern', methods=['POST'])
def modify_pattern():
    """Endpoint to inject specific bioelectric patterns into the simulation"""
    data = request.json
    pattern_type = data.get('pattern_type', 'two_head')  # e.g., 'two_head', 'eye_induction'
    region = data.get('region', [0, 0, 10, 10])  # region to modify
    intensity = data.get('intensity', 0.8)
    
    # Modify the bioelectric state in the running simulation
    models['core'].inject_bioelectric_pattern(pattern_type, region, intensity)
    
    # Return updated state
    return jsonify(get_current_state())

@app.route('/api/scenarios/planarian_regeneration', methods=['POST'])
def planarian_regeneration():
    """Simulate planarian two-headed regeneration experiment"""
    # Initialize planarian model with default one-head pattern
    init_models(planarian_config)
    
    # Apply ion channel manipulations to create two-head pattern
    models['core'].apply_ion_channel_blocker('ouabain', concentration=0.5)
    
    # Run simulation for specified time
    steps = request.json.get('steps', 100)
    for _ in range(steps):
        run_step()
    
    return jsonify({"message": "Planarian regeneration completed", 
                    "state": get_current_state()})

@app.route('/api/scenarios/<scenario_type>', methods=['POST'])
def run_experiment(scenario_type):
    """Run a predefined Levin-inspired experiment"""
    try:
        data = request.json
        steps = data.get('steps', 50)
        grid_size = default_config['core'].get('field_dimension', 10)
        
        # Initialize with appropriate state for the experiment
        if scenario_type == 'two_head_planarian':
            # Create planarian-like voltage distribution
            voltage_potential = torch.zeros((grid_size, grid_size), device=device)
            
            # Create body axis with gradient
            for i in range(grid_size):
                value = 0.8 - (i / grid_size * 1.6)  # Head is more positive
                voltage_potential[i, :] = value
            
            # Apply two positive spots for two heads
            voltage_potential[1:3, 3:7] = 0.9  # Top head
            voltage_potential[grid_size-3:grid_size-1, 3:7] = 0.9  # Bottom head
            
            description = "Two-headed planarian experiment: Manipulating bioelectric patterns to induce two-headed regeneration"
            
        elif scenario_type == 'eye_induction':
            # Create base pattern with belly region
            voltage_potential = torch.zeros((grid_size, grid_size), device=device)
            
            # Create general body pattern
            voltage_potential[:] = 0.2  # Base body voltage
            
            # Add eye-inducing voltage in belly region (center-right)
            center_x, center_y = grid_size//2, int(grid_size * 0.7)
            for i in range(grid_size):
                for j in range(grid_size):
                    # Create circular high-voltage region
                    dist = ((i - center_x)**2 + (j - center_y)**2) ** 0.5
                    if dist < grid_size * 0.15:  # Small circular region
                        voltage_potential[i, j] = 0.9  # Eye-inducing voltage
            
            description = "Ectopic Eye Formation: Inducing eye development in non-head regions by creating eye-specific voltage patterns"
            
        elif scenario_type == 'organ_reprogramming':
            # Create pattern for organ conversion (like tail-to-limb)
            voltage_potential = torch.zeros((grid_size, grid_size), device=device)
            
            # Add main body
            voltage_potential[2:grid_size-2, 2:5] = 0.3  # Body column
            
            # Add misplaced organ (tail at side)
            side_y = grid_size // 2
            # Diagonal tail coming from side
            for i in range(4):
                for j in range(3):
                    voltage_potential[side_y+i, 5+j+i] = 0.4 - (i*0.05)
            
            # Add limb-inducing signal
            for i in range(3):
                for j in range(3):
                    voltage_potential[side_y+i, 5+j] = 0.7  # Limb-inducing voltage
            
            description = "Organ Reprogramming: Converting a misplaced tail into a limb through bioelectric signaling"
            
        else:
            return jsonify({'success': False, 'error': f"Unknown scenario: {scenario_type}"})
        
        # Create standard ion gradients
        ion_gradients = {
            'sodium': torch.ones((grid_size, grid_size), device=device) * 0.1,
            'potassium': torch.ones((grid_size, grid_size), device=device) * 0.8,
            'calcium': torch.ones((grid_size, grid_size), device=device) * 0.2,
        }
        
        # Adjust ion gradients based on voltage (simplified relationship)
        for i in range(grid_size):
            for j in range(grid_size):
                v = voltage_potential[i, j].item()
                ion_gradients['sodium'][i, j] = max(0.05, min(0.3, 0.1 + v * 0.2))
                ion_gradients['potassium'][i, j] = max(0.5, min(0.95, 0.8 - v * 0.3))
                ion_gradients['calcium'][i, j] = max(0.1, min(0.4, 0.2 + v * 0.2))
        
        # Create morphological state - initially just zeros
        morphological_state = torch.zeros(grid_size*grid_size, device=device)
        
        # Run specified number of steps to evolve the pattern
        state = BioelectricState(
            voltage_potential=voltage_potential,
            ion_gradients=ion_gradients,
            gap_junction_states=torch.ones_like(voltage_potential),
            morphological_state=morphological_state
        )
        
        # Process through models for specified steps
        for _ in range(steps):
            state = models['core'](state)
            state = models['pattern_model'](state)
            state = models['colony'](state)
            state = models['homeostasis'](state)
        
        # Store result in current state
        global current_state
        current_state = {
            'voltage_potential': state.voltage_potential.cpu().numpy().tolist(),
            'ion_gradients': {
                ion: grad.cpu().numpy().tolist() for ion, grad in state.ion_gradients.items()
            },
            'morphological_state': state.morphological_state.cpu().numpy().tolist()
        }
        
        return jsonify({
            'success': True,
            'description': description,
            'state': current_state,
            'iteration': steps
        })
    
    except Exception as e:
        logger.error(f"Error running experiment {scenario_type}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/modify_cell', methods=['POST'])
def modify_cell():
    """Modify a specific cell in the bioelectric grid."""
    try:
        data = request.json
        x = data.get('x', 0)
        y = data.get('y', 0)
        ion_type = data.get('ion', 'voltage')
        intensity = data.get('intensity', 0.8)
        
        # Get global state
        global current_state
        
        if current_state is None:
            return jsonify({'success': False, 'error': 'No active simulation'})
        
        # Modify the appropriate property
        if ion_type == 'voltage':
            voltage_arr = current_state['voltage_potential']
            if x < len(voltage_arr) and y < len(voltage_arr[0]):
                voltage_arr[x][y] = intensity
        else:
            # Modify ion gradient
            if ion_type in current_state['ion_gradients']:
                ion_arr = current_state['ion_gradients'][ion_type]
                if x < len(ion_arr) and y < len(ion_arr[0]):
                    ion_arr[x][y] = intensity
        
        return jsonify({'success': True, 'state': current_state})
    
    except Exception as e:
        logger.error(f"Error modifying cell: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

def convert_to_plotly_format(data, data_type='voltage'):
    """
    Convert tensor data to format suitable for plotly visualization
    
    Args:
        data: Tensor data (voltage, ion gradient, etc.)
        data_type: Type of data ('voltage', 'ion', 'morphology')
        
    Returns:
        dict: Plot data for plotly
    """
    import numpy as np
    
    # Convert tensor to numpy if needed
    if hasattr(data, 'cpu'):
        data_np = data.cpu().numpy()
    elif isinstance(data, list):
        data_np = np.array(data)
    else:
        data_np = data
        
    # Set colorscale based on data type
    if data_type == 'voltage':
        colorscale = 'Viridis'
        title = 'Membrane Potential (mV)'
    elif data_type == 'sodium':
        colorscale = 'Hot'
        title = 'Sodium Concentration (mM)'
    elif data_type == 'potassium':
        colorscale = 'Blues'
        title = 'Potassium Concentration (mM)'
    elif data_type == 'calcium':
        colorscale = 'Greens'
        title = 'Calcium Concentration (mM)'
    else:  # morphology
        colorscale = 'Cividis'
        title = 'Morphological State'
    
    # Create plot data
    plot_data = {
        'z': data_np.tolist() if isinstance(data_np, np.ndarray) else data_np,
        'type': 'heatmap',
        'colorscale': colorscale,
        'colorbar': {'title': title}
    }
    
    return plot_data

@app.route('/api/test', methods=['GET'])
def api_test():
    """Simple endpoint to test API connectivity."""
    return jsonify({
        'success': True,
        'message': 'API is working!',
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

if __name__ == '__main__':
    app.run(debug=True)