import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import json
import pathlib
import plotly.graph_objects as go
import plotly.express as px

# Add project root to path for imports
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

# Import our project modules
from core.models.coherence_model import CoherenceModel
from core.simulators.coherence_simulator import CoherenceSimulator
from api.llm_client import LLMClient
from app.explanations.concept_definitions import get_concept_explanation

# Import Basic Mode interface
from app.dashboard.basic_mode import show_basic_mode_interface

# Add imports for new modules
from app.translation.parameter_mapping import ParameterTranslator
from app.templates.experiment_templates import ExperimentTemplate, get_all_templates, get_template_by_name
from app.visualization.basic_visualizations import plot_with_interpretations, create_summary_card

# Setup page config
st.set_page_config(
    page_title="Reflective Coherence Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for storing experiment results
if 'experiments' not in st.session_state:
    st.session_state.experiments = {}
if 'current_experiment' not in st.session_state:
    st.session_state.current_experiment = None
if 'llm_insights' not in st.session_state:
    st.session_state.llm_insights = {}
# Add interface mode to session state
if 'interface_mode' not in st.session_state:
    st.session_state.interface_mode = "select"

# Initialize simulator
@st.cache_resource
def get_simulator():
    return CoherenceSimulator()

simulator = get_simulator()

# Initialize LLM client if API keys are available
@st.cache_resource
def get_llm_client():
    openai_key = os.environ.get("OPENAI_API_KEY")
    claude_key = os.environ.get("CLAUDE_API_KEY")
    
    if openai_key or claude_key:
        return LLMClient(
            openai_api_key=openai_key,
            claude_api_key=claude_key,
            default_provider="openai" if openai_key else "claude"
        )
    return None

llm_client = get_llm_client()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        color: #0D47A1;
    }
    .concept {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .parameter-slider {
        margin-bottom: 0.5rem;
    }
    .mode-switch {
        background-color: #F5F5F5;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .stRadio > div {
        display: flex;
        justify-content: center;
    }
    .mode-container {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .mode-button {
        background-color: #f0f7ff;
        border: 1px solid #ddd;
        padding: 10px 20px;
        margin: 0 10px;
        border-radius: 5px;
        cursor: pointer;
        text-align: center;
    }
    .mode-button.active {
        background-color: #1E88E5;
        color: white;
        border: 1px solid #1976D2;
    }
</style>
""", unsafe_allow_html=True)

# Define custom entropy functions
def constant_entropy(t, value=0.3):
    return value

def increasing_entropy(t, start=0.1, rate=0.005):
    return start + rate * t

def decreasing_entropy(t, start=0.5, rate=0.005):
    return max(0.1, start - rate * t)

def oscillating_entropy(t, base=0.3, amplitude=0.15, frequency=0.1):
    return base + amplitude * np.sin(frequency * t)

# Function to render the mode selection interface
def render_mode_selection():
    """Render the mode selection interface with Basic and Expert options."""
    st.write("# Reflective Coherence Explorer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("## Expert Mode")
        st.write("Complete control over all scientific parameters with access to the full range of experimental options and detailed analysis tools.")
        if st.button("Use Expert Mode"):
            st.session_state.interface_mode = "expert"
            st.experimental_rerun()
    
    with col2:
        st.write("## Basic Mode")
        st.write("Simplified interface with intuitive controls, pre-configured templates, and enhanced visualizations with interpretations.")
        if st.button("Use Basic Mode"):
            st.session_state.interface_mode = "basic"
            st.experimental_rerun()
    
    st.markdown("---")

@st.cache_data
def run_expert_simulation(experiment_id, alpha, beta, K, initial_coherence, entropy_fn, time_span, description):
    """
    Run experiment simulation with caching for improved performance.
    
    Args:
        experiment_id: Unique identifier for the experiment
        alpha: Coherence growth rate
        beta: Entropy influence
        K: Maximum coherence
        initial_coherence: Initial coherence value
        entropy_fn: Function to calculate entropy
        time_span: Time points for simulation
        description: Description of the experiment
        
    Returns:
        Simulation results
    """
    simulator = get_simulator()
    
    # Configure the experiment
    simulator.configure_experiment(
        experiment_id=experiment_id,
        alpha=alpha,
        beta=beta,
        K=K,
        initial_coherence=initial_coherence,
        entropy_fn=entropy_fn,
        description=description,
        time_span=time_span
    )
    
    # Run the experiment
    results = simulator.run_experiment(experiment_id)
    
    # Downsample large datasets for more efficient visualization
    if len(results['time']) > 1000:
        from app.visualization.basic_visualizations import downsample_time_series
        
        time_ds, coherence_ds = downsample_time_series(results['time'], results['coherence'], 1000)
        _, entropy_ds = downsample_time_series(results['time'], results['entropy'], 1000)
        
        # Only downsample threshold if it's an array
        if isinstance(results['threshold'], np.ndarray):
            _, threshold_ds = downsample_time_series(results['time'], results['threshold'], 1000)
        else:
            threshold_ds = results['threshold']
        
        # Update results with downsampled data
        results.update({
            'time': time_ds,
            'coherence': coherence_ds,
            'entropy': entropy_ds,
            'threshold': threshold_ds
        })
    
    return results

@st.cache_data
def generate_expert_visualization(_experiment_id, results):
    """
    Generate visualizations for expert mode with caching.
    
    Args:
        _experiment_id: Used for cache invalidation when experiment changes
        results: Experiment results data
        
    Returns:
        Plotly figure for interactive visualization
    """
    # Create visualization (this can be expensive for large datasets)
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Add coherence trace
    fig.add_trace(go.Scatter(
        x=results['time'],
        y=results['coherence'],
        mode='lines',
        name='Coherence',
        line=dict(color='blue', width=2)
    ))
    
    # Add entropy trace
    fig.add_trace(go.Scatter(
        x=results['time'],
        y=results['entropy'],
        mode='lines',
        name='Entropy',
        line=dict(color='red', width=2)
    ))
    
    # Add threshold
    if isinstance(results['threshold'], np.ndarray):
        fig.add_trace(go.Scatter(
            x=results['time'],
            y=results['threshold'],
            mode='lines',
            name='Threshold',
            line=dict(color='green', width=1, dash='dash')
        ))
    else:
        # Constant threshold
        fig.add_shape(
            type="line",
            x0=min(results['time']),
            y0=results['threshold'],
            x1=max(results['time']),
            y1=results['threshold'],
            line=dict(color="green", width=1, dash="dash"),
        )
        
        # Add as trace for legend
        fig.add_trace(go.Scatter(
            x=[min(results['time']), max(results['time'])],
            y=[results['threshold'], results['threshold']],
            mode='lines',
            name='Threshold',
            line=dict(color='green', width=1, dash='dash')
        ))
    
    # Update layout
    fig.update_layout(
        title=f"Coherence Dynamics",
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Metrics",
        hovermode="x unified",
        template="plotly_white"
    )
    
    return fig

# Render the Expert Mode sidebar controls
def render_expert_sidebar():
    st.sidebar.markdown('<div class="main-header">Controls</div>', unsafe_allow_html=True)

    # Experiment configuration
    st.sidebar.markdown('<div class="sub-header">Experiment Parameters</div>', unsafe_allow_html=True)

    # Get experiment name
    experiment_id = st.sidebar.text_input("Experiment Name", value="Experiment_1")

    # Parameter sliders
    alpha = st.sidebar.slider("Coherence Growth Rate (α)", 0.01, 0.5, 0.1, 0.01, 
                            help="Controls how quickly coherence accumulates")
    beta = st.sidebar.slider("Entropy Influence (β)", 0.0, 1.0, 0.2, 0.05,
                          help="Controls how strongly entropy impacts coherence")
    K = st.sidebar.slider("Maximum Coherence (K)", 0.5, 2.0, 1.0, 0.1,
                        help="The maximum possible coherence value")
    initial_coherence = st.sidebar.slider("Initial Coherence", 0.1, 0.9, 0.5, 0.1,
                                        help="Starting coherence value")

    # Entropy function selection
    entropy_type = st.sidebar.selectbox(
        "Entropy Function",
        ["Random", "Constant", "Increasing", "Decreasing", "Oscillating"],
        help="Pattern of entropy over time"
    )

    # Additional parameters based on entropy type
    if entropy_type == "Constant":
        entropy_value = st.sidebar.slider("Entropy Value", 0.1, 0.9, 0.3, 0.05)
        entropy_fn = lambda t: constant_entropy(t, entropy_value)
    elif entropy_type == "Increasing":
        entropy_start = st.sidebar.slider("Starting Entropy", 0.1, 0.5, 0.1, 0.05)
        entropy_rate = st.sidebar.slider("Increase Rate", 0.001, 0.01, 0.005, 0.001)
        entropy_fn = lambda t: increasing_entropy(t, entropy_start, entropy_rate)
    elif entropy_type == "Decreasing":
        entropy_start = st.sidebar.slider("Starting Entropy", 0.2, 0.9, 0.5, 0.05)
        entropy_rate = st.sidebar.slider("Decrease Rate", 0.001, 0.01, 0.005, 0.001)
        entropy_fn = lambda t: decreasing_entropy(t, entropy_start, entropy_rate)
    elif entropy_type == "Oscillating":
        entropy_base = st.sidebar.slider("Base Entropy", 0.1, 0.5, 0.3, 0.05)
        entropy_amplitude = st.sidebar.slider("Amplitude", 0.05, 0.3, 0.15, 0.05)
        entropy_frequency = st.sidebar.slider("Frequency", 0.01, 0.5, 0.1, 0.01)
        entropy_fn = lambda t: oscillating_entropy(t, entropy_base, entropy_amplitude, entropy_frequency)
    else:  # Random
        entropy_fn = None  # Use default random entropy

    # Time parameters
    simulation_time = st.sidebar.slider("Simulation Time", 10, 500, 100, 10)
    time_steps = st.sidebar.slider("Time Steps", 100, 1000, 500, 100)
    time_span = np.linspace(0, simulation_time, time_steps)

    # Description for experiment
    description = st.sidebar.text_area(
        "Experiment Description",
        value=f"Testing coherence dynamics with {entropy_type.lower()} entropy",
        height=100
    )

    # Configure and run experiment
    if st.sidebar.button("Run Experiment"):
        with st.spinner("Running experiment..."):
            # Use cached run function
            results = run_expert_simulation(
                experiment_id,
                alpha,
                beta,
                K,
                initial_coherence,
                entropy_fn,
                time_span,
                description
            )
            
            # Store in session state
            st.session_state.experiments[experiment_id] = results
            st.session_state.current_experiment = experiment_id
            
            # Generate LLM insights if available
            if llm_client and llm_client.is_available():
                try:
                    with st.spinner("Generating insights with AI..."):
                        insights = llm_client.generate_insight(
                            results,
                            specificity="general"
                        )
                    st.session_state.llm_insights[experiment_id] = insights
                except Exception as e:
                    st.error(f"Error generating AI insights: {e}")
            
            st.success(f"Experiment '{experiment_id}' completed successfully!")

    # Compare experiments section
    st.sidebar.markdown('<div class="sub-header">Compare Experiments</div>', unsafe_allow_html=True)

    # Only show if we have experiments to compare
    if st.session_state.experiments:
        experiment_options = list(st.session_state.experiments.keys())
        selected_experiments = st.sidebar.multiselect(
            "Select Experiments to Compare",
            options=experiment_options,
            default=[],
            help="Choose multiple experiments to compare their results"
        )
        
        compare_parameter = st.sidebar.radio(
            "Compare Parameter",
            ["coherence", "entropy"],
            help="Which parameter to compare across experiments"
        )
        
        if st.sidebar.button("Compare Selected Experiments") and len(selected_experiments) > 1:
            with st.spinner("Generating comparison..."):
                # Check if all experiments are in simulator's records
                valid_experiments = [exp for exp in selected_experiments if exp in simulator.results]
                if len(valid_experiments) < 2:
                    st.warning("Need at least 2 valid experiments for comparison. Some experiments may not have been run in this session.")
                else:
                    simulator.compare_experiments(valid_experiments, compare_parameter, save_plot=True)
                    st.success("Comparison generated!")
                    # We'll display this in the main area

    # Concept explanation section
    st.sidebar.markdown('<div class="sub-header">Explain Concepts</div>', unsafe_allow_html=True)

    # Define concept options
    concept_options = [
        "Coherence Accumulation",
        "Entropy Dynamics",
        "Phase Transition Threshold",
        "Reflective Consistency",
        "Memory Selection",
        "Self-Loss Function",
        "Gradient Descent in Coherence",
        "Multi-Agent Coherence"
    ]

    # Select concept to explain
    selected_concept = st.sidebar.selectbox(
        "Select Concept to Explain",
        options=concept_options,
        help="Get an explanation of this concept"
    )

    # Select audience level
    audience_level = st.sidebar.radio(
        "Explanation Level",
        ["beginner", "intermediate", "advanced"],
        help="Tailor the explanation to your knowledge level"
    )

    # Button to generate explanation
    if st.sidebar.button("Generate Explanation"):
        # Try to use LLM for explanation if available
        if llm_client and llm_client.is_available():
            with st.spinner(f"Generating explanation for {selected_concept}..."):
                try:
                    explanation = llm_client.explain_concept(
                        selected_concept,
                        audience_level=audience_level
                    )
                    st.session_state.concept_explanation = {
                        "concept": selected_concept,
                        "level": audience_level,
                        "text": explanation,
                        "source": "llm"
                    }
                    st.success("Explanation generated!")
                except Exception as e:
                    # Fall back to static explanation if LLM fails
                    st.warning(f"LLM explanation failed: {e}. Using pre-defined explanation instead.")
                    explanation = get_concept_explanation(selected_concept, audience_level)
                    st.session_state.concept_explanation = {
                        "concept": selected_concept,
                        "level": audience_level,
                        "text": explanation,
                        "source": "static"
                    }
        else:
            # Use static explanation if LLM not available
            explanation = get_concept_explanation(selected_concept, audience_level)
            st.session_state.concept_explanation = {
                "concept": selected_concept,
                "level": audience_level,
                "text": explanation,
                "source": "static"
            }
            st.success("Explanation loaded!")

# Render the Expert Mode main content
def render_expert_content():
    st.markdown('<div class="main-header">Reflective Coherence Explorer</div>', unsafe_allow_html=True)

    # Introduction
    with st.expander("About Reflective Coherence (ΨC)", expanded=True):
        st.markdown("""
        <div class="info-box">
        <p>Reflective Coherence (ΨC) is a mathematical framework that models how systems maintain internal consistency
        while adapting to changing conditions. The core principle suggests that coherence accumulates according to
        logistic growth, modified by entropy, with phase transitions occurring when certain thresholds are crossed.</p>
        
        <p>This application lets you explore the dynamics of coherence by setting up experiments with different parameters
        and observing how coherence evolves over time in response to varying entropy conditions.</p>
        
        <p>Use the controls in the sidebar to configure and run experiments, then analyze the results here.</p>
        </div>
        """, unsafe_allow_html=True)

    # Display current experiment if available
    if st.session_state.current_experiment:
        current_exp = st.session_state.current_experiment
        results = st.session_state.experiments[current_exp]
        
        st.markdown(f'<div class="sub-header">Results: {current_exp}</div>', unsafe_allow_html=True)
        
        # Generate visualization with caching
        fig = generate_expert_visualization(current_exp, results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Create columns for layout
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Display experiment summary
            st.markdown(f"### {results['interpretation']['phase_state'].capitalize()} System")
            st.markdown(f"**Description:** {results['description']}")
            
            # Key metrics
            st.markdown("### Key Metrics")
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.metric(
                    "Final Coherence", 
                    f"{results['results']['coherence']['end']:.3f}",
                    f"{results['results']['coherence']['change']:.3f}"
                )
                st.metric(
                    "Threshold",
                    f"{results['results']['threshold']:.3f}"
                )
            
            with metrics_col2:
                st.metric(
                    "Mean Entropy",
                    f"{results['results']['entropy']['mean']:.3f}"
                )
                st.metric(
                    "Parameters",
                    f"α={results['parameters']['alpha']}, β={results['parameters']['beta']}"
                )
            
            # Key findings
            st.markdown("### Interpretation")
            st.info(results['interpretation']['key_finding'])
            
            # LLM insights if available
            if current_exp in st.session_state.llm_insights:
                insights = st.session_state.llm_insights[current_exp]
                st.markdown("### AI Insights")
                st.markdown(f"**Headline:** {insights['headline']}")
                with st.expander("View Full Analysis"):
                    st.markdown(insights['analysis'])

    # Display comparison if available
    if 'selected_experiments' in locals() and len(selected_experiments) > 1:
        valid_experiments = [exp for exp in selected_experiments if exp in simulator.results]
        if len(valid_experiments) >= 2:
            st.markdown('<div class="sub-header">Experiment Comparison</div>', unsafe_allow_html=True)
            
            # Generate comparison plot
            fig = simulator.compare_experiments(valid_experiments, compare_parameter, save_plot=False)
            st.pyplot(fig)
            
            # Display parameter comparison table
            st.markdown("### Parameter Comparison")
            
            # Create dataframe for comparison
            comparison_data = {}
            for exp_id in valid_experiments:
                exp_results = st.session_state.experiments[exp_id]
                comparison_data[exp_id] = {
                    "α (alpha)": exp_results['parameters']['alpha'],
                    "β (beta)": exp_results['parameters']['beta'],
                    "K": exp_results['parameters']['K'],
                    "Initial Coherence": exp_results['parameters']['initial_coherence'],
                    "Final Coherence": exp_results['results']['coherence']['end'],
                    "Mean Entropy": exp_results['results']['entropy']['mean'],
                    "Threshold": exp_results['results']['threshold'],
                    "Phase State": exp_results['interpretation']['phase_state'].capitalize()
                }
            
            comparison_df = pd.DataFrame(comparison_data).T
            st.dataframe(comparison_df)

    # Display concept explanation if available
    if 'concept_explanation' in st.session_state:
        concept_info = st.session_state.concept_explanation
        st.markdown(f'<div class="sub-header">Understanding {concept_info["concept"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<p><em>Explanation level: {concept_info["level"]}</em></p>', unsafe_allow_html=True)
        
        # Display the source of the explanation (LLM or static)
        if concept_info.get("source") == "llm":
            st.markdown('<p><em>Explanation generated by AI</em></p>', unsafe_allow_html=True)
        else:
            st.markdown('<p><em>Pre-defined explanation</em></p>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="concept">
        {concept_info["text"]}
        </div>
        """, unsafe_allow_html=True)

# Function to render the Expert Mode interface
def render_expert_interface():
    """Render the expert interface with sidebar controls and main content area."""
    render_expert_sidebar()
    render_expert_content()

# Function to render the Basic Mode interface
def render_basic_interface(simulator):
    st.markdown('<div class="sub-header">Reflective Coherence Explorer - Basic Mode</div>', unsafe_allow_html=True)
    
    # Create main columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Experiment Setup")
        
        # Option to use a template or custom settings
        setup_option = st.radio(
            "Setup Method",
            ["Use Template", "Custom Setup"],
            help="Choose a predefined template or create your own experiment"
        )
        
        experiment_id = st.text_input("Experiment Name", "Basic_Experiment")
        
        if setup_option == "Use Template":
            # Display available templates
            templates = get_all_templates()
            template_names = [t.name for t in templates]
            
            selected_template_name = st.selectbox(
                "Select Experiment Template",
                options=template_names,
                help="Choose a predefined experiment that demonstrates a specific coherence phenomenon"
            )
            
            selected_template = get_template_by_name(selected_template_name)
            
            # Display template description
            st.info(selected_template.description)
            
            # Option to view scientific parameters
            with st.expander("View Scientific Parameters"):
                st.write("Alpha:", selected_template.parameters["alpha"])
                st.write("Beta:", selected_template.parameters["beta"])
                st.write("K:", selected_template.parameters["K"])
                st.write("Initial Coherence:", selected_template.parameters["initial_coherence"])
                st.write("Entropy Function:", selected_template.parameters["entropy_fn"].__name__ if callable(selected_template.parameters["entropy_fn"]) else "Default")
            
            # Allow minimal customization
            simulation_time = st.slider("Simulation Duration", 10, 300, int(selected_template.parameters["time_span"][-1]), 10)
            
            # Run experiment button
            if st.button("Run Experiment"):
                with st.spinner("Running template experiment..."):
                    # Apply template to simulator
                    selected_template.apply_to_simulator(
                        simulator,
                        experiment_id=experiment_id,
                        override_parameters={"time_span": np.linspace(0, simulation_time, 500)}
                    )
                    
                    # Run the experiment
                    results = simulator.run_experiment(experiment_id)
                    
                    # Get summary
                    summary = simulator.get_experiment_summary(experiment_id)
                    
                    # Store in session state
                    st.session_state.experiments[experiment_id] = summary
                    st.session_state.current_experiment = experiment_id
                    
                    st.success(f"Template experiment '{experiment_id}' completed!")
        
        else:  # Custom Setup
            st.markdown("#### Simplified Parameters")
            
            # Simplified parameters using translator
            growth_speed = st.slider("Growth Speed", 0.1, 10.0, 5.0, 0.1,
                                    help="How quickly coherence builds up (higher = faster)")
            
            uncertainty_impact = st.slider("Uncertainty Impact", 0.1, 10.0, 5.0, 0.1,
                                         help="How strongly uncertainty affects coherence (higher = stronger impact)")
            
            system_capacity = st.slider("System Capacity", 1.0, 10.0, 5.0, 0.5,
                                      help="Maximum potential coherence the system can achieve")
            
            starting_stability = st.slider("Starting Stability", 0.1, 1.0, 0.3, 0.1,
                                         help="Initial coherence level (higher = more stable start)")
            
            # Entropy type selection with simpler options
            entropy_type = st.selectbox(
                "Uncertainty Pattern",
                ["Stable", "Increasing", "Decreasing", "Fluctuating"],
                help="Pattern of uncertainty over time"
            )
            
            # Simplified entropy parameters based on type
            if entropy_type == "Stable":
                entropy_level = st.slider("Uncertainty Level", 0.1, 1.0, 0.5, 0.1)
                entropy_params = {"value": entropy_level}
            elif entropy_type == "Increasing":
                entropy_start = st.slider("Starting Uncertainty", 0.1, 0.5, 0.2, 0.1)
                entropy_rate = st.slider("Rate of Increase", 0.001, 0.01, 0.005, 0.001)
                entropy_params = {"start": entropy_start, "rate": entropy_rate}
            elif entropy_type == "Decreasing":
                entropy_start = st.slider("Starting Uncertainty", 0.5, 1.0, 0.8, 0.1)
                entropy_rate = st.slider("Rate of Decrease", 0.001, 0.01, 0.005, 0.001)
                entropy_params = {"start": entropy_start, "rate": entropy_rate}
            else:  # Fluctuating
                entropy_base = st.slider("Base Uncertainty", 0.1, 0.5, 0.3, 0.1)
                entropy_change = st.slider("Amount of Fluctuation", 0.05, 0.3, 0.15, 0.05)
                entropy_speed = st.slider("Speed of Fluctuation", 0.01, 0.5, 0.1, 0.01)
                entropy_params = {"base": entropy_base, "amplitude": entropy_change, "frequency": entropy_speed}
            
            # Time parameters
            simulation_time = st.slider("Simulation Duration", 10, 300, 100, 10)
            
            # Run experiment button
            if st.button("Run Custom Experiment"):
                with st.spinner("Running custom experiment..."):
                    # Translate parameters to expert mode
                    expert_params = ParameterTranslator.basic_to_expert(
                        growth_speed, 
                        uncertainty_impact,
                        system_capacity,
                        starting_stability
                    )
                    
                    # Translate entropy parameters
                    entropy_fn = ParameterTranslator.translate_entropy_params(entropy_type, entropy_params)
                    
                    # Configure experiment with translated parameters
                    simulator.configure_experiment(
                        experiment_id=experiment_id,
                        alpha=expert_params["alpha"],
                        beta=expert_params["beta"],
                        K=expert_params["K"],
                        initial_coherence=expert_params["initial_coherence"],
                        entropy_fn=entropy_fn,
                        time_span=np.linspace(0, simulation_time, 500),
                        description=f"Basic Mode custom experiment with {entropy_type.lower()} uncertainty"
                    )
                    
                    # Run experiment
                    results = simulator.run_experiment(experiment_id)
                    
                    # Get summary
                    summary = simulator.get_experiment_summary(experiment_id)
                    
                    # Store in session state
                    st.session_state.experiments[experiment_id] = summary
                    st.session_state.current_experiment = experiment_id
                    
                    st.success(f"Custom experiment '{experiment_id}' completed!")
    
    # Render results in the second column if available
    with col2:
        if st.session_state.current_experiment:
            current_exp = st.session_state.current_experiment
            summary = st.session_state.experiments[current_exp]
            
            st.markdown(f"### Results: {current_exp}")
            
            # Use enhanced visualizations from basic_visualizations module
            fig = plot_with_interpretations(
                simulator.results[current_exp]["time"],
                simulator.results[current_exp]["coherence"],
                simulator.results[current_exp]["entropy"],
                summary["results"]["threshold"]
            )
            st.pyplot(fig)
            
            # Generate and display summary card
            summary_card = create_summary_card(
                simulator.results[current_exp]["time"],
                simulator.results[current_exp]["coherence"],
                simulator.results[current_exp]["entropy"],
                summary["results"]["threshold"]
            )
            
            # Display user-friendly interpretation
            st.markdown("### What This Means")
            st.markdown(f"**System Status:** {summary_card['status']}")
            st.markdown(f"**Stability Level:** {summary_card['stability_level']}")
            st.info(summary_card['interpretation'])
            
            # Option to view scientific details
            with st.expander("View Scientific Details"):
                st.write("Final Coherence:", f"{summary['results']['coherence']['end']:.3f}")
                st.write("Mean Entropy:", f"{summary['results']['entropy']['mean']:.3f}")
                st.write("Threshold:", f"{summary['results']['threshold']:.3f}")
                st.write("Alpha:", f"{summary['parameters']['alpha']}")
                st.write("Beta:", f"{summary['parameters']['beta']}")
                st.write("K:", f"{summary['parameters']['K']}")
                
                # Display technical interpretation
                st.markdown("#### Technical Interpretation")
                st.write(summary['interpretation']['key_finding'])
                
                # Show phase transitions if detected
                if 'phase_transitions' in summary_card:
                    st.markdown("#### Phase Transitions")
                    for pt in summary_card['phase_transitions']:
                        st.write(f"At t={pt['time']:.2f}: {pt['description']}")

# Main function to render the appropriate interface
def main():
    """Main entry point for the dashboard application."""
    # Setup page config
    st.set_page_config(
        page_title="Reflective Coherence Explorer",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Initialize session state
    if 'coherence_simulator' not in st.session_state:
        st.session_state.coherence_simulator = CoherenceSimulator()
    
    if 'experiment_results' not in st.session_state:
        st.session_state.experiment_results = {}
    
    if 'interface_mode' not in st.session_state:
        st.session_state.interface_mode = "select"  # Default to mode selection
    
    # Determine which interface to show
    if st.session_state.interface_mode == "select":
        render_mode_selection()
    elif st.session_state.interface_mode == "expert":
        render_expert_interface()
    elif st.session_state.interface_mode == "basic":
        show_basic_mode_interface()
    
    # Add a way to return to mode selection
    if st.session_state.interface_mode != "select":
        if st.sidebar.button("Switch Interface Mode"):
            st.session_state.interface_mode = "select"
            st.experimental_rerun()
    
    # Check for API keys warning
    if not llm_client:
        st.sidebar.markdown("""
        <div class="warning-box">
        <b>Note:</b> No API keys found for OpenAI or Claude.
        Set environment variables <code>OPENAI_API_KEY</code> and/or 
        <code>CLAUDE_API_KEY</code> to enable AI insights and explanations.
        </div>
        """, unsafe_allow_html=True)
    
    # Footer with information
    st.markdown("---")
    st.markdown("""
    <p style="text-align: center; color: #666;">
    Reflective Coherence Explorer | ΨC Principle Visualization Tool
    </p>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main() 