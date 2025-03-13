"""
Visualization utilities for bioelectric states and patterns.
Provides tools to visualize bioelectric potentials, cell colonies,
pattern formation, and other BCM components.
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

# Define custom colormaps for bioelectric visualization
BIOELECTRIC_CMAP = LinearSegmentedColormap.from_list(
    'bioelectric', 
    ['darkblue', 'blue', 'lightblue', 'white', 'yellow', 'orange', 'red'],
    N=256
)

def plot_voltage_potential(state: BioelectricState, 
                          ax: Optional[plt.Axes] = None,
                          title: str = "Bioelectric Potential") -> plt.Axes:
    """
    Plot the voltage potential field
    
    Args:
        state: BioelectricState containing voltage potential
        ax: Optional matplotlib axis to plot on
        title: Plot title
        
    Returns:
        Matplotlib axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    # Get voltage data
    voltage = state.voltage_potential.detach().cpu().numpy()
    
    # Plot as both a line and a heatmap
    x = np.arange(len(voltage))
    
    # Plot voltage line
    ax.plot(x, voltage, 'k-', linewidth=1.5, alpha=0.7)
    
    # Create heatmap below the line
    extent = [0, len(voltage), np.min(voltage) - 0.5, np.max(voltage) + 0.5]
    voltage_grid = np.tile(voltage, (50, 1))
    ax.imshow(voltage_grid, aspect='auto', cmap=BIOELECTRIC_CMAP, 
              alpha=0.7, extent=extent, origin='lower')
    
    # Add some styling
    ax.set_title(title)
    ax.set_xlabel("Spatial Position")
    ax.set_ylabel("Potential")
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    norm = plt.Normalize(vmin=np.min(voltage), vmax=np.max(voltage))
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=BIOELECTRIC_CMAP), 
                 cax=cax, label="Voltage")
    
    return ax

def visualize_ion_gradients(state: BioelectricState, 
                           fig: Optional[plt.Figure] = None) -> plt.Figure:
    """
    Visualize ion gradients across the bioelectric field
    
    Args:
        state: BioelectricState containing ion gradients
        fig: Optional matplotlib figure to plot on
        
    Returns:
        Matplotlib figure with the plots
    """
    if fig is None:
        fig = plt.figure(figsize=(12, 8))
    
    # Get ion data
    ion_names = list(state.ion_gradients.keys())
    n_ions = len(ion_names)
    
    # Create subplot grid
    grid_size = int(np.ceil(np.sqrt(n_ions)))
    
    for i, ion_name in enumerate(ion_names):
        ax = fig.add_subplot(grid_size, grid_size, i+1)
        ion_data = state.ion_gradients[ion_name].detach().cpu().numpy()
        
        # Plot ion gradient
        x = np.arange(len(ion_data))
        ax.plot(x, ion_data, '-', linewidth=2, label=ion_name)
        ax.set_title(f"{ion_name} Gradient")
        ax.set_xlabel("Spatial Position")
        ax.set_ylabel("Concentration")
        ax.legend()
        
        # Add shaded region for visual emphasis
        ax.fill_between(x, 0, ion_data, alpha=0.3)
        
    fig.tight_layout()
    return fig

def plot_cell_colony(colony_data: Dict, 
                    fig: Optional[plt.Figure] = None, 
                    plot_connections: bool = True,
                    threshold: float = 0.3) -> plt.Figure:
    """
    Visualize a cell colony with its connections and states
    
    Args:
        colony_data: Dictionary containing positions, connections, and states
        fig: Optional matplotlib figure to plot on
        plot_connections: Whether to plot connections between cells
        threshold: Minimum connection strength to plot
        
    Returns:
        Matplotlib figure with the colony visualization
    """
    if fig is None:
        fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    
    positions = colony_data['positions']
    connections = colony_data['connections']
    states = colony_data['states']
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes with positions
    for i, pos in enumerate(positions):
        G.add_node(i, pos=pos)
    
    # Add edges with weights
    if plot_connections:
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                if connections[i, j] > threshold:
                    G.add_edge(i, j, weight=connections[i, j])
    
    # Get node positions
    node_pos = nx.get_node_attributes(G, 'pos')
    
    # Calculate node colors based on mean activation
    node_colors = [np.mean(states[i]) for i in range(len(positions))]
    
    # Draw the network
    edges = G.edges()
    weights = [G[u][v]['weight'] * 3 for u, v in edges]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, node_pos, node_size=800, 
                          node_color=node_colors, cmap=BIOELECTRIC_CMAP,
                          alpha=0.9, ax=ax)
    
    # Draw edges
    if plot_connections and edges:
        nx.draw_networkx_edges(G, node_pos, edgelist=edges, width=weights, 
                              edge_color='gray', alpha=0.6, ax=ax)
    
    # Add node labels
    nx.draw_networkx_labels(G, node_pos, font_size=10, 
                           font_family='sans-serif', ax=ax)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=BIOELECTRIC_CMAP, 
                              norm=plt.Normalize(vmin=min(node_colors), 
                                               vmax=max(node_colors)))
    sm._A = []  # Hack to make it work with non-array data
    cbar = plt.colorbar(sm, ax=ax, label='Cell Activation')
    
    # Remove axis
    ax.set_axis_off()
    ax.set_title("Cell Colony Network")
    
    plt.tight_layout()
    return fig

def plot_pattern_formation(initial_pattern: np.ndarray, 
                          final_pattern: np.ndarray,
                          metrics: Dict,
                          fig: Optional[plt.Figure] = None) -> plt.Figure:
    """
    Visualize pattern formation process and metrics
    
    Args:
        initial_pattern: Starting pattern
        final_pattern: Pattern after formation process
        metrics: Dictionary of pattern metrics
        fig: Optional matplotlib figure
        
    Returns:
        Matplotlib figure with the visualization
    """
    if fig is None:
        fig = plt.figure(figsize=(12, 6))
    
    # Plot initial and final patterns
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(initial_pattern, 'b-', label='Initial')
    ax1.set_title("Initial Pattern")
    ax1.set_xlabel("Position")
    ax1.set_ylabel("Value")
    
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(final_pattern, 'r-', label='Final')
    ax2.set_title("Final Pattern")
    ax2.set_xlabel("Position")
    ax2.set_ylabel("Value")
    
    # Plot pattern change
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(initial_pattern, 'b-', alpha=0.5, label='Initial')
    ax3.plot(final_pattern, 'r-', alpha=0.5, label='Final')
    ax3.fill_between(range(len(initial_pattern)), 
                    initial_pattern, final_pattern, 
                    color='purple', alpha=0.2)
    ax3.set_title("Pattern Change")
    ax3.set_xlabel("Position")
    ax3.set_ylabel("Value")
    ax3.legend()
    
    # Plot metrics
    ax4 = fig.add_subplot(2, 2, 4)
    metrics_to_plot = {k: v for k, v in metrics.items() 
                      if isinstance(v, (int, float))}
    
    if metrics_to_plot:
        y_pos = np.arange(len(metrics_to_plot))
        values = list(metrics_to_plot.values())
        labels = list(metrics_to_plot.keys())
        
        bars = ax4.barh(y_pos, values, align='center')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(labels)
        ax4.set_title("Pattern Metrics")
        ax4.set_xlim(0, max(1.0, max(values) * 1.1))  # Ensure scale is at least 0-1
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', ha='left', va='center')
            
    fig.tight_layout()
    return fig

def create_bioelectric_animation(state_history: List[BioelectricState], 
                                filename: str = "bioelectric_animation.mp4",
                                interval: int = 200) -> str:
    """
    Create an animation of bioelectric state changes over time
    
    Args:
        state_history: List of BioelectricState objects over time
        filename: Output filename
        interval: Time between frames in milliseconds
        
    Returns:
        Path to saved animation file
    """
    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract voltage data
    voltage_data = [state.voltage_potential.detach().cpu().numpy() 
                   for state in state_history]
    
    # Find global min/max for consistent colormap
    v_min = min([np.min(v) for v in voltage_data])
    v_max = max([np.max(v) for v in voltage_data])
    norm = plt.Normalize(v_min, v_max)
    
    # Initial plot
    line, = ax.plot(voltage_data[0], 'k-', lw=2)
    img = ax.imshow(np.tile(voltage_data[0], (50, 1)), 
                   aspect='auto', cmap=BIOELECTRIC_CMAP, 
                   alpha=0.7, origin='lower', 
                   extent=[0, len(voltage_data[0]), v_min, v_max])
    
    # Add colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=BIOELECTRIC_CMAP), 
                       ax=ax, label="Voltage")
    
    # Title with frame counter
    title = ax.set_title("Bioelectric State: Frame 0")
    
    def update(frame):
        """Update function for animation"""
        voltage = voltage_data[frame]
        line.set_ydata(voltage)
        img.set_data(np.tile(voltage, (50, 1)))
        title.set_text(f"Bioelectric State: Frame {frame}")
        return line, img, title
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(voltage_data),
                                blit=True, interval=interval)
    
    # Save animation
    writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='BCM'),
                                  bitrate=1800)
    ani.save(filename, writer=writer)
    
    plt.close()
    return filename

def save_state_visualization(state: BioelectricState, 
                            metrics: Dict, 
                            filename: str) -> str:
    """
    Save visualization of bioelectric state to file
    
    Args:
        state: BioelectricState to visualize
        metrics: Dictionary of metrics to display
        filename: Output filename
        
    Returns:
        Path to saved visualization
    """
    fig = plt.figure(figsize=(12, 10))
    
    # Plot voltage potential
    ax1 = fig.add_subplot(2, 2, 1)
    plot_voltage_potential(state, ax=ax1)
    
    # Plot ion gradients (simplified to fit this subplot)
    ax2 = fig.add_subplot(2, 2, 2)
    ion_names = list(state.ion_gradients.keys())
    for ion_name in ion_names:
        ion_data = state.ion_gradients[ion_name].detach().cpu().numpy()
        ax2.plot(ion_data, label=ion_name)
    ax2.set_title("Ion Gradients")
    ax2.set_xlabel("Spatial Position")
    ax2.set_ylabel("Concentration")
    ax2.legend()
    
    # Plot morphological state
    ax3 = fig.add_subplot(2, 2, 3)
    morphology = state.morphological_state.detach().cpu().numpy()
    ax3.plot(morphology)
    ax3.set_title("Morphological State")
    ax3.set_xlabel("Component")
    ax3.set_ylabel("Value")
    
    # Plot metrics
    ax4 = fig.add_subplot(2, 2, 4)
    # Filter for scalar metrics
    scalar_metrics = {k: v for k, v in metrics.items() 
                     if isinstance(v, (int, float))}
    
    if scalar_metrics:
        # Create bar chart of metrics
        y_pos = np.arange(len(scalar_metrics))
        values = list(scalar_metrics.values())
        labels = list(scalar_metrics.keys())
        
        bars = ax4.barh(y_pos, values, align='center')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(labels)
        ax4.set_title("Bioelectric Metrics")
        ax4.set_xlim(0, max(1.0, max(values) * 1.1))
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', ha='left', va='center')
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return filename

def plot_multi_state_comparison(states: List[BioelectricState],
                               labels: List[str],
                               filename: Optional[str] = None) -> plt.Figure:
    """
    Compare multiple bioelectric states side by side
    
    Args:
        states: List of BioelectricState objects to compare
        labels: Labels for each state
        filename: Optional filename to save the plot
        
    Returns:
        Matplotlib figure with the comparison
    """
    fig, axes = plt.subplots(len(states), 1, figsize=(10, 3*len(states)))
    if len(states) == 1:
        axes = [axes]
        
    for ax, state, label in zip(axes, states, labels):
        voltage = state.voltage_potential.detach().cpu().numpy()
        ax.plot(voltage, 'k-', linewidth=1.5)
        ax.set_title(f"{label}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Voltage")
        
        # Add heatmap visualization below the line
        extent = [0, len(voltage), np.min(voltage) - 0.5, np.max(voltage) + 0.5]
        voltage_grid = np.tile(voltage, (20, 1))
        ax.imshow(voltage_grid, aspect='auto', cmap=BIOELECTRIC_CMAP, 
                 alpha=0.7, extent=extent, origin='lower')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        
    return fig

def plot_goal_signals(goal_signals: torch.Tensor,
                     goal_names: List[str] = None,
                     ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Visualize goal activation signals
    
    Args:
        goal_signals: Tensor of goal activation values
        goal_names: Optional list of goal names
        ax: Optional matplotlib axis
        
    Returns:
        Matplotlib axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        
    # Convert to numpy
    if isinstance(goal_signals, torch.Tensor):
        goal_values = goal_signals.detach().cpu().numpy()
    else:
        goal_values = np.array(goal_signals)
        
    # Create goal names if not provided
    if goal_names is None:
        goal_names = [f"Goal {i+1}" for i in range(len(goal_values))]
        
    # Create horizontal bar chart
    y_pos = np.arange(len(goal_values))
    
    # Sort by value for better visualization
    sorted_indices = np.argsort(goal_values)
    sorted_values = goal_values[sorted_indices]
    sorted_names = [goal_names[i] for i in sorted_indices]
    
    # Create bars with gradient colors
    cmap = plt.cm.get_cmap('viridis')
    colors = [cmap(v/max(1.0, max(sorted_values))) for v in sorted_values]
    
    bars = ax.barh(y_pos, sorted_values, align='center', color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.set_title("Goal Activation Signals")
    ax.set_xlabel("Activation Strength")
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{width:.2f}', ha='left', va='center')
        
    return ax

def create_state_heatmap(state_matrix: np.ndarray,
                        row_labels: List[str],
                        col_labels: List[str],
                        title: str = "State Heatmap",
                        filename: Optional[str] = None) -> plt.Figure:
    """
    Create a heatmap visualization of state data
    
    Args:
        state_matrix: 2D numpy array of state values
        row_labels: Labels for rows (e.g., components)
        col_labels: Labels for columns (e.g., time points)
        title: Plot title
        filename: Optional filename to save the plot
        
    Returns:
        Matplotlib figure with the heatmap
    """
    # Set up figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(state_matrix, annot=False, cmap=BIOELECTRIC_CMAP,
               linewidths=0.5, ax=ax)
    
    # Add labels
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticklabels(row_labels, rotation=0)
    
    ax.set_title(title)
    
    # Add colorbar label
    cbar = ax.collections[0].colorbar
    cbar.set_label('Value')
    
    # Save if filename provided
    if filename:
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        
    return fig