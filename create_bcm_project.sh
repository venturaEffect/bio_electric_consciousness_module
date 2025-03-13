echo "Creating Bioelectric Consciousness Module project structure..."

# Create directory structure
mkdir -p configs
mkdir -p docs
mkdir -p bcm/{__pycache__,core,morphology,collective,goals,memory,bridge,utils}
mkdir -p bcm/{core,morphology,collective,goals,memory,bridge,utils}/__pycache__
mkdir -p simulations/{environments,scenarios,evaluation}
mkdir -p simulations/{environments,scenarios,evaluation}/__pycache__
mkdir -p experiments
mkdir -p scripts
mkdir -p tests
mkdir -p output/visualizations
mkdir -p data/{raw,processed}

# Create __init__.py files for all Python packages
find . -type d -name "bcm*" -o -name "simulations*" -o -name "tests" | while read dir; do
    touch "$dir/__init__.py"
    echo "Created: $dir/__init__.py"
done

# Create README.md
cat > README.md << 'EOF'
# Bioelectric Consciousness Module (BCM)

## ⚠️ Important Disclaimer
This project is **NOT** affiliated with or endorsed by Dr. Michael Levin or his research group. The Bioelectric Consciousness Module (BCM) is inspired by scientific concepts from Dr. Levin's work but represents our own interpretation and implementation for research and learning purposes only.

## Overview
The Bioelectric Consciousness Module (BCM) is an experimental project exploring the computational implementation of primitive consciousness based on bioelectric principles observed in simple cellular organisms. This project aims to develop a foundation for artificial consciousness that begins at the cellular level, focusing on:

- Bioelectric signaling networks and voltage-mediated information processing
- Collective cellular intelligence through gap junction communication
- Primitive goal-directed behavior and homeostasis
- Pattern formation and morphological computation

## Inspiration & Scientific Foundation
This project draws inspiration from the pioneering work of Dr. Michael Levin and colleagues on bioelectricity in morphogenesis, regeneration, and primitive cognition. Their research suggests that bioelectric signaling may underpin information processing at cellular levels, forming a kind of "primitive cognition" that doesn't require neural systems.

Key scientific concepts explored include:
- Voltage-mediated pattern storage and processing
- Gap junction-mediated collective behavior
- Bioelectric encoding of goal states
- Non-neural memory and decision making

## Project Goals
1. Create a computational model of bioelectric information processing
2. Demonstrate primitive goal-directed behavior in simulated cellular systems
3. Develop a bridge to more complex consciousness models (ACM)
4. Explore the developmental basis of consciousness from simple to complex systems

## Getting Started
```bash
# Clone the repository
git clone https://github.com/yourusername/bioelectric-consciousness-module.git
cd bioelectric-consciousness-module

# Install dependencies
pip install -r [requirements.txt](http://_vscodecontentref_/0)

# Run a basic simulation
python scripts/run_simulation.py --config configs/bioelectric_config.yaml