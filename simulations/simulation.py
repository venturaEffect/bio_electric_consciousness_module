"""
Main simulation module for bioelectric consciousness.
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time

from bcm.core.bioelectric_core import BioelectricConsciousnessCore, BioelectricState
from bcm.morphology.pattern_formation import BioelectricPatternFormation
from bcm.collective.cell_colony import CellColony
from bcm.goals.homeostasis import HomeostasisRegulator
from bcm.utils.metrics import calculate_pattern_stability
from simulations.environments.env_setup import BioelectricEnvironment

class BioelectricSimulation:
    """
    Main simulation class for bioelectric consciousness experiments.
    Integrates all components and runs the simulation loop.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 sim_params: Dict[str, Any],
                 device: str = "cpu"):
        """
        Initialize the simulation with configuration parameters.
        
        Args:
            config: Configuration dictionary for BCM components
            sim_params: Simulation parameters
            device: Device to run the simulation on ('cpu' or 'cuda')
        """
        self.config = config
        self.sim_params = sim_params
        self.device = device
        self.state_history = []
        
        # Initialize components
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize all components required for the simulation."""
        # Environment
        env_size = self.sim_params['environment']['size']
        self.environment = BioelectricEnvironment(
            size=tuple(env_size),
            boundary_condition=self.sim_params['environment']['boundary_condition'],
            noise_level=self.sim_params['environment']['noise_level']
        )
        
        # Core BCM components
        self.bcm_core = BioelectricConsciousnessCore(
            input_dim=self.config['core']['input_dim'],
            hidden_dim=self.config['core']['hidden_dim'],
            output_dim=self.config['core']['output_dim'],
            num_layers=self.config['core']['num_layers']
        ).to(self.device)
        
        # Pattern formation
        self.pattern_model = BioelectricPatternFormation(
            spatial_resolution=self.config['morphology']['spatial_resolution'],
            diffusion_rate=self.config['morphology']['diffusion_rate'],
            reaction_rate=self.config['morphology']['reaction_rate']
        ).to(self.device)
        
        # Cell colony
        self.colony = CellColony(
            colony_size=self.config['collective']['colony_size'],
            connection_density=self.config['collective']['connection_density']
        ).to(self.device)
        
        # Homeostasis regulator
        self.homeostasis = HomeostasisRegulator(
            strength=self.config['goals']['homeostasis_strength'],
            adaptation_rate=self.config['goals']['adaptation_rate']
        ).to(self.device)
    
    def initialize_state(self) -> BioelectricState:
        """
        Initialize the simulation state.
        
        Returns:
            Initial bioelectric state
        """
        # Initialize with environment information
        env_state = self.environment.get_state()
        
        # Convert numpy arrays to torch tensors
        torch_state = {
            key: torch.tensor(value, dtype=torch.float32, device=self.device)
            for key, value in env_state.items() 
            if isinstance(value, np.ndarray)
        }
        
        # Handle nested dictionaries like ion_gradients
        if 'ion_gradients' in env_state:
            torch_state['ion_gradients'] = {
                ion: torch.tensor(gradient, dtype=torch.float32, device=self.device)
                for ion, gradient in env_state['ion_gradients'].items()
            }
        
        # Create initial BioelectricState
        initial_state = BioelectricState()  # This would need proper initialization based on your BioelectricState class
        
        return initial_state
        
    def run(self, time_steps: Optional[int] = None) -> List[BioelectricState]:
        """
        Run the simulation for the specified number of time steps.
        
        Args:
            time_steps: Number of time steps to run (defaults to config value)
            
        Returns:
            List of bioelectric states over time
        """
        if time_steps is None:
            time_steps = self.sim_params['simulation']['time_steps']
            
        dt = self.sim_params['simulation']['dt']
        save_interval = self.sim_params['simulation']['save_interval']
        
        # Initialize state
        current_state = self.initialize_state()
        self.state_history = [current_state]
        
        # Main simulation loop
        start_time = time.time()
        for t in range(time_steps):
            # Update environment
            self.environment.update()
            
            # Process through bioelectric core
            current_state = self.bcm_core(current_state)
            
            # Form patterns
            current_state = self.pattern_model(current_state)
            
            # Apply colony interactions
            current_state = self.colony(current_state)
            
            # Apply homeostasis regulation
            current_state = self.homeostasis(current_state)
            
            # Apply environmental influence
            env_state = self.environment.get_state()
            # This would depend on how your BioelectricState class handles environmental updates
            
            # Save state at intervals
            if t % save_interval == 0:
                self.state_history.append(current_state)
                
                # Calculate metrics for monitoring
                stability = calculate_pattern_stability(current_state)
                
                # Print progress
                elapsed = time.time() - start_time
                print(f"Step {t}/{time_steps} | Stability: {stability:.4f} | Time: {elapsed:.2f}s")
        
        return self.state_history