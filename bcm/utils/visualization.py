"""
Visualization tools for bioelectric consciousness module.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
import os
from pathlib import Path
import seaborn as sns
from ..core.bioelectric_core import BioelectricState
from matplotlib.animation import FuncAnimation

# Define custom colormaps for bioelectric visualization
BIOELECTRIC_CMAP = LinearSegmentedColormap.from_list(
    'bioelectric', 
    ['darkblue', 'blue', 'lightblue', 'white', 'yellow', 'orange', 'red'],
    N=256
)

def plot_voltage_potential(state, save_path=None, show=False):
    """Plot voltage potential field from a bioelectric state."""
    if isinstance(state.voltage_potential, torch.Tensor):
        voltage = state.voltage_potential.detach().cpu().numpy()
    else:
        voltage = state.voltage_potential
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(voltage.reshape(10, -1), cmap='viridis', annot=False)
    plt.title('Bioelectric Voltage Potential')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    if show:
        plt.show()
    else:
        plt.close()

def visualize_ion_gradients(state, save_path=None, show=False):
    """Visualize ion gradients from a bioelectric state."""
    ion_names = list(state.ion_gradients.keys())
    num_ions = len(ion_names)
    
    fig, axes = plt.subplots(1, num_ions, figsize=(5*num_ions, 5))
    
    for i, ion in enumerate(ion_names):
        if isinstance(state.ion_gradients[ion], torch.Tensor):
            gradient = state.ion_gradients[ion].detach().cpu().numpy()
        else:
            gradient = state.ion_gradients[ion]
            
        sns.heatmap(gradient.reshape(10, -1), ax=axes[i], cmap='plasma')
        axes[i].set_title(f'{ion} gradient')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_cell_colony(state, save_path=None, show=False):
    """Plot cell colony from a bioelectric state."""
    # Implementation depends on how cell colony is represented
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Placeholder implementation
    return None

def create_bioelectric_animation(state_history, save_path=None, show=False):
    """Create an animation of bioelectric state changes over time."""
    # Basic placeholder implementation
    if not state_history:
        return None
        
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def update(frame):
        ax.clear()
        state = state_history[frame]
        if isinstance(state.voltage_potential, torch.Tensor):
            voltage = state.voltage_potential.detach().cpu().numpy()
        else:
            voltage = state.voltage_potential
        sns.heatmap(voltage.reshape(10, -1), ax=ax, cmap='viridis', annot=False)
        ax.set_title(f'Frame {frame}')
        
    anim = FuncAnimation(fig, update, frames=min(30, len(state_history)), interval=200)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        anim.save(save_path, writer='ffmpeg', fps=5)
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_pattern_formation(morphological_state, time_steps=None, save_path=None, show=False):
    """
    Plot pattern formation over time or at a specific time step.
    
    Args:
        morphological_state: Either a single state or a list of states over time
        time_steps: Time points for the states (if applicable)
        save_path: Path to save the visualization
        show: Whether to display the plot
    """
    plt.figure(figsize=(12, 8))
    
    if isinstance(morphological_state, list):
        # Plot pattern evolution over time
        if time_steps is None:
            time_steps = list(range(len(morphological_state)))
            
        for i, state in enumerate(morphological_state):
            if isinstance(state, torch.Tensor):
                state_np = state.detach().cpu().numpy()
            else:
                state_np = state
                
            # Reshape to 2D if needed for visualization
            if len(state_np.shape) == 1:
                side_length = int(np.sqrt(state_np.shape[0]))
                if side_length**2 == state_np.shape[0]:  # Perfect square
                    state_np = state_np.reshape(side_length, side_length)
                else:
                    # Just create a roughly square shape
                    side1 = int(np.sqrt(state_np.shape[0]))
                    side2 = state_np.shape[0] // side1
                    state_np = state_np.reshape(side1, side2)
            
            plt.subplot(1, len(morphological_state), i+1)
            sns.heatmap(state_np, cmap='viridis', annot=False)
            plt.title(f'Time: {time_steps[i]}')
    else:
        # Plot a single state
        if isinstance(morphological_state, torch.Tensor):
            state_np = morphological_state.detach().cpu().numpy()
        else:
            state_np = morphological_state
            
        # Reshape to 2D if needed
        if len(state_np.shape) == 1:
            side_length = int(np.sqrt(state_np.shape[0]))
            if side_length**2 == state_np.shape[0]:  # Perfect square
                state_np = state_np.reshape(side_length, side_length)
            else:
                # Just create a roughly square shape
                side1 = int(np.sqrt(state_np.shape[0]))
                side2 = state_np.shape[0] // side1
                state_np = state_np.reshape(side1, side2)
        
        sns.heatmap(state_np, cmap='viridis', annot=False)
        plt.title('Morphological Pattern')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    if show:
        plt.show()
    else:
        plt.close()