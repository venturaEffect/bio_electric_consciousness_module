from flask import Flask, render_template, request, jsonify
import os
import sys
import torch
import numpy as np
import yaml

# Add parent directory to path so we can import BCM modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import BCM modules
from bcm.core.bioelectric_core import BioelectricConsciousnessCore
from bcm.collective.cell_colony import CellColony
from bcm.morphology.pattern_formation import BioelectricPatternFormation
from bcm.goals.homeostasis import HomeostasisRegulator
from bcm.utils.visualization import (
    plot_voltage_potentials, 
    plot_ion_gradients, 
    plot_morphological_state
)

# Create Flask app
app = Flask(__name__)

# Load default configuration
config_path = os.path.join(parent_dir, 'configs', 'bioelectric_config.yaml')
with open(config_path, 'r') as f:
    default_config = yaml.safe_load(f)

# Global variables
models = {}
current_state = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_models(config):
    """Initialize models based on configuration."""
    global models
    
    models['core'] = BioelectricConsciousnessCore(config['core']).to(device)
    models['pattern_model'] = BioelectricPatternFormation(config['pattern_formation']).to(device)
    models['colony'] = CellColony(config['colony']).to(device)
    models['homeostasis'] = HomeostasisRegulator(config['homeostasis']).to(device)
    
    return models

@app.route('/')
def index():
    """Main page with interactive simulation interface."""
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
    data = request.json
    config_updates = data.get('config_updates', {})
    
    # Update config with provided parameters
    updated_config = default_config.copy()
    for section, params in config_updates.items():
        if section in updated_config:
            updated_config[section].update(params)
    
    # Re-initialize models if config changed significantly
    init_models(updated_config)
    
    # Convert state back to tensors for processing
    global current_state
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
        current_state = {
            'voltage_potential': state.voltage_potential.cpu().numpy().tolist(),
            'ion_gradients': {
                ion: grad.cpu().numpy().tolist() for ion, grad in state.ion_gradients.items()
            },
            'morphological_state': state.morphological_state.cpu().numpy().tolist()
        }
        
        return jsonify({
            'success': True,
            'state': current_state
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

if __name__ == '__main__':
    app.run(debug=True)