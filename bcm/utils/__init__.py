"""
Utility functions for the Bioelectric Consciousness Module.
"""
from bcm.utils.metrics import (
    calculate_pattern_stability,
    calculate_bioelectric_entropy,
    calculate_spatial_coherence,
    calculate_goal_alignment,
    summarize_bioelectric_state
)
from bcm.utils.visualization import (
    plot_voltage_potential,
    visualize_ion_gradients,
    plot_cell_colony,
    plot_pattern_formation,
    create_bioelectric_animation
)

__all__ = [
    'calculate_pattern_stability',
    'calculate_bioelectric_entropy',
    'calculate_spatial_coherence',
    'calculate_goal_alignment',
    'summarize_bioelectric_state',
    'plot_voltage_potential',
    'visualize_ion_gradients',
    'plot_cell_colony',
    'plot_pattern_formation',
    'create_bioelectric_animation'
]