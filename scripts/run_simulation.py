#!/usr/bin/env python
"""
Run a bioelectric consciousness simulation with the specified configuration.
"""
import argparse
import yaml
import torch
import os
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path to import BCM
sys.path.append(str(Path(__file__).resolve().parent.parent))

from bcm.core.bioelectric_core import BioelectricConsciousnessCore, BioelectricState
from bcm.morphology.pattern_formation import BioelectricPatternFormation
from bcm.collective.cell_colony import CellColony
from bcm.goals.homeostasis import HomeostasisRegulator
from bcm.goals.goal_states import GoalState
from bcm.utils.visualization import (
    plot_voltage_potential,
    visualize_ion_gradients,
    plot_cell_colony,
    create_bioelectric_animation
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run a bioelectric consciousness simulation")
    parser.add_argument("--config", type=str, default="configs/bioelectric_config.yaml",
                        help="Path to the bioelectric configuration file")
    parser.add_argument("--sim-params", type=str, default="configs/simulation_params.yaml",
                        help="Path to the simulation parameters file")
    parser.add_argument("--output-dir", type=str, default="output/",
                        help="Directory to save outputs and visualizations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run simulation on (cuda/cpu)")
    return parser.parse_args()

def load_configs(args):
    with open(args.config, 'r') as f:
        bioelectric_config = yaml.safe_load(f)
    with open(args.sim_params, 'r') as f:
        sim_params = yaml.safe_load(f)
    return bioelectric_config, sim_params

def initialize_models(config, device):
    # Core config remains the same...
    core_config = {
        'field_dimension': config['core']['output_dim'],
        'sodium_channel': {
            'input_dim': config['core']['input_dim'],
            'hidden_dim': config['core']['hidden_dim'],
            'output_dim': config['core']['output_dim']
        },
        'potassium_channel': {
            'input_dim': config['core']['input_dim'],
            'hidden_dim': config['core']['hidden_dim'],
            'output_dim': config['core']['output_dim']
        },
        'calcium_channel': {
            'input_dim': config['core']['input_dim'],
            'hidden_dim': config['core']['hidden_dim'],
            'output_dim': config['core']['output_dim']
        },
        'morphology': {
            'hidden_dim': config['core']['hidden_dim'],
            'state_dim': config['core']['output_dim'] // 2,
        },
        'goals': {
            'hidden_dim': config['core']['hidden_dim'],
            'num_goals': 4
        }
    }
    
    # Initialize with the structured config
    core = BioelectricConsciousnessCore(config=core_config).to(device)
    
    # Pattern formation config
    pattern_config = {
        'field_dimension': config['core']['output_dim'],
        'pattern_complexity': config['morphology'].get('pattern_complexity', 0.7),
        'reaction_rate': config['morphology'].get('reaction_rate', 0.1),
        'morphology': {
            'stability_preference': 0.7,
            'plasticity': 0.3
        }
    }
    pattern_model = BioelectricPatternFormation(config=pattern_config).to(device)
    
    # Create a proper config for cell colony that INCLUDES the sodium_channel structure
    colony_config = {
        'field_dimension': config['core']['output_dim'],
        'colony_size': config['collective'].get('colony_size', 10),
        'gap_junction_conductance': config['collective'].get('gap_junction_conductance', 0.2),
        # Include the required channel configs to avoid KeyError
        'sodium_channel': core_config['sodium_channel'],
        'potassium_channel': core_config['potassium_channel'],
        'calcium_channel': core_config['calcium_channel'],
        'morphology': core_config['morphology'],
        'goals': core_config['goals']
    }
    colony = CellColony(config=colony_config).to(device)
    
    homeostasis = HomeostasisRegulator(
        config={
            'homeostasis_strength': config['goals'].get('homeostasis_strength', 0.8),
            'resting_potential': 0.2,
            'field_dimension': config['core']['output_dim']
        }
    ).to(device)
    
    return {
        'core': core,
        'pattern_model': pattern_model,
        'colony': colony,
        'homeostasis': homeostasis
    }

def run_simulation(models, sim_params, output_dir, device):
    # Setup simulation parameters
    time_steps = sim_params['simulation']['time_steps']
    dt = sim_params['simulation']['dt']
    save_interval = sim_params['simulation']['save_interval']
    
    # Initialize state history for visualization
    state_history = []
    
    # Create a properly initialized BioelectricState
    field_dim = models['core'].config['field_dimension']
    initial_state = BioelectricState(
        voltage_potential=torch.zeros((10, 10), device=device),
        ion_gradients={
            'sodium': torch.ones((10, 10), device=device) * 0.1,
            'potassium': torch.ones((10, 10), device=device) * 0.9,
            'calcium': torch.ones((10, 10), device=device) * 0.2
        },
        gap_junction_states=torch.eye(10, device=device),
        morphological_state=torch.zeros(field_dim//2, device=device)
    )
    
    current_state = initial_state
    
    # Run simulation
    for t in range(time_steps):
        # Process through bioelectric core
        current_state = models['core'](current_state)
        
        # Form patterns
        current_state = models['pattern_model'](current_state)
        
        # Apply colony interactions
        current_state = models['colony'](current_state)
        
        # Apply homeostasis regulation
        current_state = models['homeostasis'](current_state)
        
        # Save state at intervals
        if t % save_interval == 0:
            state_history.append(current_state)
            print(f"Timestep {t}/{time_steps} completed.")
    
    # Create visualizations
    create_bioelectric_animation(state_history, save_path=f"{output_dir}/animation.mp4")
    
    # Plot final state
    plot_voltage_potential(state_history[-1], save_path=f"{output_dir}/final_voltage.png")
    
    return state_history

def main():
    args = parse_args()
    bioelectric_config, sim_params = load_configs(args)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(sim_params['simulation']['random_seed'])
    np.random.seed(sim_params['simulation']['random_seed'])
    
    # Initialize models
    models = initialize_models(bioelectric_config, args.device)
    
    # Run simulation
    state_history = run_simulation(models, sim_params, args.output_dir, args.device)
    
    print(f"Simulation completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()