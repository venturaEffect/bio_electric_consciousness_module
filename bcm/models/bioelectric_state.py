# (this is an example path, use your actual path)

class BioelectricState:
    """Container for bioelectric simulation state."""
    
    def __init__(self, voltage_potential, ion_gradients, gap_junction_states, morphological_state):
        """Initialize a new bioelectric state."""
        self.voltage_potential = voltage_potential
        self.ion_gradients = ion_gradients
        self.gap_junction_states = gap_junction_states
        self.morphological_state = morphological_state