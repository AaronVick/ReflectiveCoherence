# Reflective Coherence Explorer User Guide

This comprehensive guide explains all the features of the Reflective Coherence Explorer in detail. It will help you navigate the interface, understand each component, and make the most of the application's capabilities.

## Table of Contents
- [Interface Overview](#interface-overview)
- [Parameter Configuration](#parameter-configuration)
- [Entropy Functions](#entropy-functions)
- [Running Experiments](#running-experiments)
- [Understanding Results](#understanding-results)
- [Comparing Experiments](#comparing-experiments)
- [Concept Explanations](#concept-explanations)
- [AI-Enhanced Features](#ai-enhanced-features)
- [Saving and Exporting](#saving-and-exporting)
- [Advanced Techniques](#advanced-techniques)

## Interface Overview

The Reflective Coherence Explorer interface is divided into two main sections:

1. **Sidebar (Control Panel)**: Located on the left side, this is where you configure experiments, select options, and trigger actions.

2. **Main Panel (Results Area)**: The larger area on the right where results, visualizations, and explanations are displayed.

[*Screenshot: Full application interface with sidebar and main panel*]

The application follows a workflow:
1. Configure parameters and settings in the sidebar
2. Run experiments using the action buttons
3. View and analyze results in the main panel

## Parameter Configuration

The sidebar contains several parameter sliders that control the core mathematical model:

### Core Parameters

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| Coherence Growth Rate | α (alpha) | 0.01-0.5 | Controls how quickly coherence accumulates |
| Entropy Influence | β (beta) | 0.0-1.0 | Controls how strongly entropy impacts coherence |
| Maximum Coherence | K | 0.5-2.0 | The maximum possible coherence value |
| Initial Coherence | C₀ | 0.1-0.9 | Starting coherence value |

[*Screenshot: Parameter sliders section*]

**Tips for Parameter Selection**:
- **α (alpha)**: Higher values lead to faster coherence growth
  - Low (0.01-0.05): Very slow growth, suitable for stable systems
  - Medium (0.05-0.2): Moderate growth, typical of most systems
  - High (0.2-0.5): Rapid growth, representing highly adaptive systems

- **β (beta)**: Controls sensitivity to entropy
  - Low (0.0-0.2): System is resistant to entropy
  - Medium (0.2-0.5): Balanced sensitivity
  - High (0.5-1.0): System is highly sensitive to entropy

- **K**: Maximum coherence capacity
  - Lower values create systems with limited capacity
  - Higher values allow for greater maximum coherence

- **Initial Coherence**: Starting point for the simulation
  - Low values (0.1-0.3): System starts mostly incoherent
  - Mid values (0.4-0.6): System starts partially coherent
  - High values (0.7-0.9): System starts highly coherent

## Entropy Functions

The entropy function determines how uncertainty or disorder evolves over time in your simulated system. The application offers several entropy patterns:

### Available Entropy Functions

| Function | Description | Real-world Analogy |
|----------|-------------|-------------------|
| Random | Unpredictable entropy values | Chaotic, unpredictable environment |
| Constant | Stable, unchanging entropy | Steady, predictable environment |
| Increasing | Gradually rising entropy | Growing complexity or deteriorating order |
| Decreasing | Declining entropy | Learning or adaptation process |
| Oscillating | Cyclical entropy patterns | Periodic environmental challenges |

[*Screenshot: Entropy function selection dropdown*]

Each entropy function has its own set of adjustable parameters:

#### Constant Entropy
- **Entropy Value**: The fixed level of entropy (0.1-0.9)

#### Increasing Entropy
- **Starting Entropy**: Initial entropy value (0.1-0.5)
- **Increase Rate**: How quickly entropy rises (0.001-0.01)

#### Decreasing Entropy
- **Starting Entropy**: Initial entropy value (0.2-0.9)
- **Decrease Rate**: How quickly entropy falls (0.001-0.01)

#### Oscillating Entropy
- **Base Entropy**: Average entropy level (0.1-0.5)
- **Amplitude**: Size of oscillations (0.05-0.3)
- **Frequency**: Speed of oscillations (0.01-0.5)

[*Screenshot: Entropy function parameter sliders*]

**Choosing the Right Entropy Function**:
- **Constant**: Good for baseline experiments and understanding basic coherence dynamics
- **Increasing**: Useful for testing system resilience and identifying tipping points
- **Decreasing**: Models learning systems or environments becoming more orderly
- **Oscillating**: Excellent for observing phase transitions and system responsiveness
- **Random**: Stress-tests the system with unpredictable challenges

## Running Experiments

To run an experiment with your selected parameters:

1. **Name Your Experiment**: Enter a descriptive name in the "Experiment Name" field. Using meaningful names helps when comparing multiple experiments later.

2. **Add a Description** (optional): Enter notes about your experiment setup or hypotheses in the "Experiment Description" field.

3. **Adjust Time Settings** (if needed):
   - **Simulation Time**: Total duration of the simulation (10-500 time units)
   - **Time Steps**: Number of calculation points (100-1000 steps)

4. **Click "Run Experiment"**: This button triggers the simulation with your current settings.

[*Screenshot: Experiment setup and run button*]

During the simulation, a progress indicator will appear. For most parameter combinations, simulations complete within a few seconds.

**Best Practices**:
- Start with shorter simulations (100 time units) to quickly test parameter effects
- Use longer simulations (300+ time units) when studying long-term behavior
- Include key parameter values in your experiment name for easier identification
- Run a "baseline" experiment with moderate parameters as a reference point

## Understanding Results

After running an experiment, results appear in the main panel. The results display includes several components:

### Coherence Plot

The central visualization shows how coherence and entropy evolve over time:

[*Screenshot: Results plot with labeled components*]

- **Blue Line**: Coherence trajectory over time
- **Red Line**: Entropy values over time
- **Green Dashed Line**: Coherence threshold (θ)

The coherence threshold divides the graph into two regions:
- **Above Threshold**: Coherent state (system has internal consistency)
- **Below Threshold**: Incoherent state (system lacks internal consistency)

### Key Metrics

Alongside the plot, key metrics summarize the experiment results:

[*Screenshot: Metrics panel*]

- **Final Coherence**: The coherence value at the end of the simulation
- **Coherence Change**: How much coherence increased or decreased over the simulation
- **Threshold**: The calculated boundary between coherent and incoherent states
- **Mean Entropy**: Average entropy throughout the simulation
- **Parameters**: A reminder of the α and β values used

### Interpretation

The system automatically provides a brief interpretation of the results:

[*Screenshot: Interpretation box*]

This includes:
- Whether the system ended in a coherent or incoherent state
- A brief explanation of what affected the coherence trajectory
- Significant observations about system behavior

## Comparing Experiments

One of the most powerful features is the ability to compare multiple experiments:

1. **Run Multiple Experiments**: First, run several experiments with different parameters.

2. **Select Experiments to Compare**: In the sidebar under "Compare Experiments," select two or more experiments from the dropdown list.

3. **Choose Comparison Parameter**: Select whether to compare "coherence" or "entropy" trajectories.

4. **Click "Compare Selected Experiments"**: Generate the comparison visualization.

[*Screenshot: Experiment comparison selection*]

The comparison appears in the main panel as:

1. **Comparison Plot**: Shows the selected parameter (coherence or entropy) for all selected experiments on the same graph, with different colors for each experiment.

2. **Parameter Comparison Table**: A data table showing all key parameters and results side by side.

[*Screenshot: Comparison results*]

**Effective Comparison Strategies**:
- **Vary One Parameter**: Change only one parameter between experiments to isolate its effect
- **Parameter Sweeps**: Run a series of experiments with gradually increasing values of one parameter
- **Entropy Function Comparison**: Compare how different entropy patterns affect the same system
- **Initial Conditions Test**: Vary only the initial coherence to test path dependency

## Concept Explanations

The Reflective Coherence Explorer includes educational features to help understand the underlying mathematics:

1. **Select Concept**: In the sidebar, choose a concept from the dropdown list, such as "Coherence Accumulation" or "Phase Transition Threshold."

2. **Choose Explanation Level**:
   - **Beginner**: Simple analogies and non-technical language
   - **Intermediate**: Some mathematical terms with intuitive explanations
   - **Advanced**: Full mathematical formulation with technical details

3. **Click "Generate Explanation"**: View the explanation in the main panel.

[*Screenshot: Concept explanation selection*]

The explanation appears in a dedicated section in the main panel, formatted for readability with key points highlighted.

[*Screenshot: Example of a concept explanation*]

**Available Concepts**:
- Coherence Accumulation
- Entropy Dynamics
- Phase Transition Threshold
- Reflective Consistency
- Memory Selection
- Self-Loss Function
- Gradient Descent in Coherence
- Multi-Agent Coherence

## AI-Enhanced Features

When API keys are configured, the application offers AI-enhanced features:

### AI Insights

After running an experiment, the system can generate AI-powered insights about your results:

[*Screenshot: AI insights section*]

These insights include:
- A headline summarizing the key finding
- Analysis of patterns and relationships in the data
- Suggestions for further experiments
- Connections to relevant theoretical concepts

### AI-Generated Explanations

When generating concept explanations with API keys configured, the explanations will be dynamically created by an AI model rather than using pre-written text. This allows for:

- More tailored explanations based on your specific interests
- Greater depth in areas of particular complexity
- More varied explanations with different analogies and perspectives

To use AI-enhanced features:
1. Set up API keys using the `setup_api_keys.py` script
2. Ensure the keys are loaded when starting the application
3. Look for the "AI Insights" section after running experiments
4. Note the "Explanation generated by AI" indicator on concept explanations

## Saving and Exporting

The Reflective Coherence Explorer automatically saves all experiment data and can export results in several ways:

### Automatic Saving

All experiment data is automatically saved to:
- **Data files**: CSV format with raw time series data
- **Metadata**: JSON format with experiment parameters and summary statistics
- **Plots**: PNG image files of visualizations

These files are stored in the `/data` directory, organized by:
- `/data/experiments/`: Raw data and metadata
- `/data/plots/`: Generated visualizations

[*Screenshot: Data directory structure*]

### Exporting Results

While viewing experiment results, you can:
- Download the raw data CSV using the download button
- Save plots as image files
- Export the parameter comparison table as CSV

## Advanced Techniques

### Custom Entropy Functions

Advanced users can define custom entropy functions by modifying the code:

1. Edit `app/dashboard/app.py`
2. Add a new function in the "Define custom entropy functions" section
3. Add your function to the dropdown options
4. Restart the application

### Extended Experiments

For deeper analysis, consider these advanced techniques:

1. **Monte Carlo Simulations**: Run multiple experiments with random parameters to identify parameter sensitivity.

2. **Bifurcation Analysis**: Systematically vary parameters to identify critical transitions in system behavior.

3. **Edge Case Testing**: Test extreme parameter values to understand system limitations.

4. **Custom Visualizations**: Export data and create specialized visualizations using external tools.

---

This guide covers the main features of the Reflective Coherence Explorer. For specific examples of experiments to run, see the [Examples and Tutorials](EXAMPLES.md) document. For technical details about the mathematical model, refer to the [Mathematical Foundation](UNDERLYING_MATH.md) document. 