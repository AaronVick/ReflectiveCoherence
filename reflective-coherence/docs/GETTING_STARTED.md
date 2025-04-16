# Getting Started with Reflective Coherence Explorer

This guide will help you set up and begin using the Reflective Coherence Explorer to understand the dynamics of coherence in systems. Whether you're a researcher, student, or simply curious about complex systems, this guide will get you up and running quickly.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Advanced Usage](#advanced-usage)
- [Understanding the Results](#understanding-the-results)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

## Prerequisites

Before you begin, ensure you have the following:

- **Python 3.8+** installed on your system
- Basic familiarity with command-line interfaces
- For AI-enhanced features: API keys from OpenAI and/or Anthropic (optional)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/reflective-coherence.git
cd reflective-coherence
```

### Step 2: Install Dependencies

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

### Step 3: Set Up API Keys (Optional)

For AI-enhanced features like concept explanations and experiment insights, you can set up API keys:

```bash
python setup_api_keys.py
```

Follow the prompts to enter your OpenAI and/or Claude API keys. The keys will be stored securely in a `.env.local` file.

*Note: The application works perfectly without API keys, but some advanced features that use large language models will be unavailable.*

## Basic Usage

### Running the Application

The simplest way to launch the application is to run:

```bash
python run.py
```

This will:
1. Check your environment for required dependencies
2. Look for your API keys (if any)
3. Start the Streamlit server
4. Open the application in your default web browser

### Your First Experiment

Once the application is running:

1. **Configure Parameters**: Use the sidebar controls to set values for:
   - α (alpha): Coherence growth rate
   - β (beta): Entropy influence
   - K: Maximum coherence
   - Initial coherence: Starting value
   - Entropy function: Pattern of entropy over time

2. **Run the Experiment**: Click the "Run Experiment" button.

3. **View Results**: The main panel will display:
   - A graph showing coherence and entropy over time
   - Key metrics including final coherence and threshold values
   - A brief interpretation of the results

## Advanced Usage

### Comparing Experiments

To understand how different parameters affect coherence dynamics:

1. Run multiple experiments with different parameter values
2. In the sidebar, select the experiments you want to compare
3. Choose which parameter to compare (coherence or entropy)
4. Click "Compare Selected Experiments"

The application will generate a comparison graph and data table showing how the different parameter sets affected the outcomes.

### Custom Entropy Functions

For more realistic simulations, you can configure different entropy patterns:

- **Constant**: Stable, unchanging entropy
- **Increasing**: Gradually rising entropy (increasing disorder)
- **Decreasing**: Declining entropy (increasing order)
- **Oscillating**: Cyclical patterns of entropy

Each pattern has adjustable parameters such as starting values, rates of change, and frequencies.

### Concept Explanations

To learn more about the mathematical concepts:

1. In the sidebar, select a concept like "Coherence Accumulation" or "Phase Transition Threshold"
2. Choose your preferred explanation level (beginner, intermediate, or advanced)
3. Click "Generate Explanation"

The explanation will be tailored to your knowledge level, from simple analogies to rigorous mathematical descriptions.

## Understanding the Results

### Key Metrics

- **Final Coherence**: The coherence value at the end of the simulation
- **Threshold (θ)**: The boundary between coherent and incoherent states
- **Mean Entropy**: Average uncertainty in the system
- **Phase State**: Whether the system ended in a coherent or incoherent state

### Interpreting the Graph

- **Blue Line**: Shows coherence evolution over time
- **Red Line**: Shows entropy over time
- **Green Dashed Line**: Indicates the coherence threshold

A system is coherent when the blue line (coherence) is above the green line (threshold).

### AI Insights

If you've set up API keys, the application can generate AI-powered insights about your experiment results, highlighting patterns and relationships that might not be immediately obvious.

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   - Error: `ModuleNotFoundError: No module named 'streamlit'`
   - Solution: Run `pip install -r requirements.txt`

2. **API Keys Not Loading**
   - Error: "No API keys found for LLM integration"
   - Solution: Run `python setup_api_keys.py` or manually create a `.env.local` file

3. **Streamlit Server Issues**
   - Error: Unable to connect to Streamlit server
   - Solution: Check if port 8501 is already in use. Try stopping other Streamlit applications or specify a different port: `streamlit run app/dashboard/app.py --server.port 8502`

4. **Visualization Problems**
   - Issue: Graphs not displaying or appearing incorrectly
   - Solution: Try adjusting your browser zoom level or use a different browser

### Getting Help

If you encounter issues not covered here, please:
1. Check the [project's GitHub repository](https://github.com/yourusername/reflective-coherence/issues) for known issues
2. Submit a new issue with details about your environment and the problem you're experiencing

## Next Steps

After getting familiar with the basic features, consider:

- **Exploring Edge Cases**: Try extreme parameter values to understand system limits
- **Designing Experiments**: Create systematic experiment series to test specific hypotheses
- **Extending the Model**: Advanced users can modify the core model in `core/models/coherence_model.py`
- **Contributing**: Share your insights, bug fixes, or feature enhancements through pull requests

---

For more detailed information about the mathematical foundations, see the [UNDERLYING_MATH.md](UNDERLYING_MATH.md) document. 