# Reflective Coherence Explorer


## In Simple Terms

The Reflective Coherence Explorer is an interactive tool that helps you understand how systems (like the human mind, organizations, or AI) maintain stability while adapting to changing environments. Think of it like a weather simulator, but instead of predicting rain or sunshine, it shows how a system's "coherence" (its internal consistency) changes when faced with uncertainty.

Imagine watching how your brain balances between:
- Sticking to what it knows (maintaining coherence)
- Adapting to new information (responding to uncertainty)

This application lets you adjust different factors and see the results through easy-to-understand visualizations. You don't need a mathematics degree to use it—the interface provides plain-language explanations of what's happening.

An interactive application for exploring the dynamics of Reflective Coherence (ΨC) - a mathematical framework for understanding how systems maintain internal consistency while adapting to changing environments.



## Overview

The Reflective Coherence Explorer allows users to:

- **Run simulations** of coherence dynamics with different parameters
- **Visualize** how coherence accumulates over time in response to entropy
- **Compare experiments** to identify patterns and relationships
- **Get AI-powered explanations** of complex mathematical concepts in plain language

This tool bridges the gap between complex mathematical theory and intuitive understanding, making the ΨC Principle accessible to users at all levels of expertise.

## Key Features

- **Interactive parameter sliders** to configure experiments
- **Real-time visualization** of coherence and entropy dynamics
- **Plain-language explanations** of results and concepts
- **Custom entropy functions** to model different environmental conditions
- **Experiment comparison** to identify patterns across parameter spaces
- **LLM integration** for AI-enhanced insights and explanations

## Documentation

- [Getting Started Guide](docs/GETTING_STARTED.md) - Installation and basic usage
- [User Guide](docs/USER_GUIDE.md) - Comprehensive guide to all features
- [Examples and Tutorials](docs/EXAMPLES.md) - Practical examples with parameter settings
- [Mathematical Foundation](docs/UNDERLYING_MATH.md) - Details of the ΨC Principle mathematics

## Quick Start (Simple Setup)

1. **Install Python**: Make sure you have [Python 3.8 or higher](https://www.python.org/downloads/) installed on your computer.

2. **Download the Application**: 
   ```bash
   git clone https://github.com/AaronVick/ReflectiveCoherence.git
   cd reflective-coherence
   ```

3. **One-Step Setup**:
   ```bash
   # This installs all requirements and sets up the application
   python setup.py
   ```

4. **Run the Application**:
   ```bash
   python run.py
   ```

5. **Open in Browser**: The application should automatically open in your web browser. If not, go to `http://localhost:8501`

## Usage

Once the application is running:

1. Use the sidebar controls to:
   - Configure experiment parameters (α, β, K, initial coherence)
   - Select entropy functions (constant, increasing, decreasing, oscillating)
   - Run simulations and compare experiments
   - Get concept explanations

2. Explore the results in the main panel:
   - Interactive visualizations of coherence and entropy over time
   - Key metrics and interpretations
   - AI-generated insights (when API keys are configured)

See the [Examples and Tutorials](docs/EXAMPLES.md) document for detailed usage scenarios.

## Understanding the Mathematical Model

The Reflective Coherence framework is built on several key mathematical components:

1. **Coherence Accumulation**: Governed by logistic growth modified by entropy.
   ```
   dC(t)/dt = α * C(t) * (1 - C(t)/K) - β * H(M(t))
   ```

2. **Entropy Dynamics**: Quantifies uncertainty in the system.
   ```
   H(M(t)) = -∑ p(m_i) * log(p(m_i))
   ```

3. **Phase Transition Threshold**: Determines when a system transitions from incoherent to coherent.
   ```
   θ = E[H(M(t))] + λθ * sqrt(Var(H(M(t))))
   ```

For a deeper dive into the mathematics, see the [Mathematical Foundation](docs/UNDERLYING_MATH.md) document.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Before contributing, please:
1. Check existing issues or create a new one describing your proposed change
2. Fork the repository and create a feature branch
3. Make your changes and add or update tests as needed
4. Ensure all tests pass with `pytest tests/`
5. Submit a pull request referencing the issue

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The mathematical foundations are based on research into reflective coherence and complex systems dynamics
- Built with Streamlit, NumPy, Pandas, and other open-source tools
- Thanks to the open-source community for providing many of the tools used in this project

## Table of Contents

- [Reflective Coherence Explorer](#reflective-coherence-explorer)
  - [In Simple Terms](#in-simple-terms)
  - [Overview](#overview)
  - [Key Features](#key-features)
  - [Documentation](#documentation)
  - [Quick Start (Simple Setup)](#quick-start-simple-setup)
  - [Usage](#usage)
  - [Understanding the Mathematical Model](#understanding-the-mathematical-model)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)
  - [Table of Contents](#table-of-contents)

## Important Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file - Project overview and setup |
| [GETTING_STARTED.md](docs/GETTING_STARTED.md) | Installation and basic usage |
| [USER_GUIDE.md](docs/USER_GUIDE.md) | Comprehensive guide to all features |
| [EXAMPLES.md](docs/EXAMPLES.md) | Practical examples with parameter settings |
| [UNDERLYING_MATH.md](docs/UNDERLYING_MATH.md) | Details of the ΨC Principle mathematics |
| [DATA_EXPORT.md](docs/DATA_EXPORT.md) | Guide to exporting and analyzing data outside the application |
| [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | Technical implementation details |
| [PARAMETER_MAPPING.md](docs/PARAMETER_MAPPING.md) | Mapping between UI parameters and mathematical model |
| [OPERATIONAL_UNDERSTANDING.md](docs/OPERATIONAL_UNDERSTANDING.md) | Comprehensive operational explanation |
| [ACCESSIBILITY_ROADMAP.md](docs/ACCESSIBILITY_ROADMAP.md) | Plan for making the tool accessible to users of all levels | 