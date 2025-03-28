from flask import Flask, render_template, request, jsonify
import os
import sys
import torch
import numpy as np
import json

# Add parent directory to path so we can import BCM modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import BCM modules
from bcm.core.bioelectric_core import BioelectricConsciousnessCore, BioelectricState
from bcm.collective.cell_colony import CellColony
from bcm.morphology.pattern_formation import BioelectricPatternFormation
from bcm.goals.homeostasis import HomeostasisRegulator
from bcm.utils.visualization import (
    plot_voltage_potentials, 
    plot_ion_gradients, 
    plot_morphological_state
)

app = Flask(__name__)

# Load default configuration
import yaml
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs/bioelectric_config.yaml'), 'r') as f:
    default_config = yaml.safe_load(f)

# Global model instances
models = {}

def init_models(config):
    """Initialize models based on configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models['core'] = BioelectricConsciousnessCore(config['core']).to(device)
    models['pattern_model'] = BioelectricPatternFormation(config['pattern_formation']).to(device)
    models['colony'] = CellColony(config['colony']).to(device)
    models['homeostasis'] = HomeostasisRegulator(config['homeostasis']).to(device)
    
    return models

# Initialize with default config
init_models(default_config)

@app.route('/')
def index():
    """Main page with simulation interface."""
    return render_template('index.html')

@app.route('/api/run_step', methods=['POST'])
def run_step():
    """Run a single simulation step with provided parameters."""
    data = request.json
    config_updates = data.get('config_updates', {})
    current_state = data.get('state', None)
    
    # Update config with provided parameters
    updated_config = default_config.copy()
    for section, params in config_updates.items():
        if section in updated_config:
            updated_config[section].update(params)
    
    # Re-initialize models if config changed significantly
    init_models(updated_config)
    
    # Create initial state if not provided
    if not current_state:
        # Create default initial state
        grid_size = updated_config['core'].get('field_dimension', 10)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize with small random values
        voltage_potential = torch.randn(grid_size, grid_size, device=device) * 0.01
        
        # Add a specific pattern for demonstration
        if data.get('scenario') == 'two_heads':
            # Add two high-voltage regions suggesting two head formation sites
            voltage_potential[1:3, 1:3] = 0.8  # Top head voltage pattern
            voltage_potential[7:9, 7:9] = 0.8  # Bottom head voltage pattern
        elif data.get('scenario') == 'eye_formation':
            # Add a voltage pattern that could induce ectopic eye
            center = grid_size // 2
            voltage_potential[center-1:center+1, center-1:center+1] = 0.9  # Central eye-inducing voltage
        
        ion_gradients = {
            'sodium': torch.randn(grid_size, grid_size, device=device) * 0.01,
            'potassium': torch.randn(grid_size, grid_size, device=device) * 0.01,
            'calcium': torch.randn(grid_size, grid_size, device=device) * 0.01,
        }
        
        morphological_state = torch.zeros(grid_size*grid_size, device=device)
        
        # Create state dictionary for JSON serialization
        current_state = {
            'voltage_potential': voltage_potential.cpu().numpy().tolist(),
            'ion_gradients': {
                ion: grad.cpu().numpy().tolist() for ion, grad in ion_gradients.items()
            },
            'morphological_state': morphological_state.cpu().numpy().tolist()
        }
    
    # Convert state back to tensors for processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    voltage_potential = torch.tensor(current_state['voltage_potential'], device=device)
    ion_gradients = {
        ion: torch.tensor(grad, device=device) 
        for ion, grad in current_state['ion_gradients'].items()
    }
    morphological_state = torch.tensor(current_state['morphological_state'], device=device)
    
    # Create BioelectricState object
    state = BioelectricState(
        voltage_potential=voltage_potential,
        ion_gradients=ion_gradients,
        gap_junction_states=torch.ones_like(voltage_potential),
        morphological_state=morphological_state
    )
    
    # Process through models
    try:
        state = models['core'](state)
        state = models['pattern_model'](state)
        state = models['colony'](state)
        state = models['homeostasis'](state)
        
        # Convert back to Python native types for JSON serialization
        result_state = {
            'voltage_potential': state.voltage_potential.cpu().numpy().tolist(),
            'ion_gradients': {
                ion: grad.cpu().numpy().tolist() for ion, grad in state.ion_gradients.items()
            },
            'morphological_state': state.morphological_state.cpu().numpy().tolist()
        }
        
        return jsonify({
            'success': True,
            'state': result_state
        })
        
    except Exception as e:
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

if __name__ == '__main__':
    app.run(debug=True)