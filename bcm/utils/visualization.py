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