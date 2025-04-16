"""
Basic Mode interface for the Reflective Coherence Explorer.

This module implements a simplified user interface for the Reflective Coherence Explorer,
making it more accessible to users without advanced mathematical knowledge. It provides
preset experiment templates, simplified parameter controls, and enhanced visualizations.

The interface uses the same underlying mathematical model as the Expert Mode but presents
it with simplified terminology and additional context.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple

# Import our modules
from app.templates.experiment_templates import get_template, get_all_templates, get_template_names
from app.translation.parameter_mapping import ParameterTranslator
from app.visualization.basic_visualizations import plot_with_interpretations, create_summary_card

def render_basic_sidebar(simulator) -> Dict[str, Any]:
    """
    Render the sidebar controls for Basic Mode.
    
    Args:
        simulator: The CoherenceSimulator instance
        
    Returns:
        Dictionary with the selected configuration or None if no experiment is run
    """
    st.sidebar.markdown('<div class="sub-header">Experiment Setup</div>', unsafe_allow_html=True)
    
    # Step 1: Template Selection
    st.sidebar.subheader("1. Choose an Experiment Template")
    template_names = get_template_names()
    selected_template_name = st.sidebar.selectbox(
        "Experiment Template", 
        template_names,
        help="Choose a pre-configured experiment with scientifically valid parameters"
    )
    
    # Get the selected template and display its description
    template = get_template(selected_template_name)
    if template:
        st.sidebar.info(template.description)
    
    # Get a experiment ID from user or generate one
    experiment_id = st.sidebar.text_input(
        "Experiment Name",
        value=f"{selected_template_name.replace(' ', '_')}_run",
        help="A unique name for this experiment run"
    )
    
    # Step 2: Optional Parameter Adjustments
    st.sidebar.subheader("2. Adjust Parameters (Optional)")
    custom_params = st.sidebar.checkbox(
        "Customize Parameters", 
        value=False,
        help="Modify the template parameters to explore different scenarios"
    )
    
    basic_config = {"experiment_id": experiment_id}
    env_params = {}
    
    if custom_params:
        with st.sidebar.expander("System Parameters"):
            # System Growth Speed (alpha)
            growth_speed = st.sidebar.select_slider(
                "System Growth Speed", 
                options=["Low", "Medium", "High"],
                value="Medium",
                help="How quickly the system builds coherence (scientific term: α)"
            )
            basic_config["System Growth Speed"] = growth_speed
            
            # Uncertainty Impact (beta)
            uncertainty_impact = st.sidebar.select_slider(
                "Uncertainty Impact", 
                options=["Low", "Medium", "High"],
                value="Medium",
                help="How strongly environmental uncertainty affects the system (scientific term: β)"
            )
            basic_config["Uncertainty Impact"] = uncertainty_impact
            
            # System Capacity (K)
            system_capacity = st.sidebar.select_slider(
                "System Capacity", 
                options=["Small", "Medium", "Large"],
                value="Medium",
                help="The maximum potential coherence the system can achieve (scientific term: K)"
            )
            basic_config["System Capacity"] = system_capacity
            
            # Starting Stability (initial_coherence)
            starting_stability = st.sidebar.select_slider(
                "Starting Stability", 
                options=["Low", "Medium", "High"],
                value="Medium",
                help="The system's initial level of coherence (scientific term: initial coherence)"
            )
            basic_config["Starting Stability"] = starting_stability
        
        with st.sidebar.expander("Environmental Pattern"):
            # Environmental Pattern (entropy function)
            env_pattern = st.sidebar.selectbox(
                "Environmental Pattern",
                [
                    "Steady Environment", 
                    "Increasing Complexity", 
                    "Learning Process", 
                    "Cyclical Challenges",
                    "Random Environment"
                ],
                help="The pattern of uncertainty in the system's environment"
            )
            basic_config["Environmental Pattern"] = env_pattern
            
            # Additional parameters based on selected pattern
            if env_pattern == "Steady Environment":
                level = st.sidebar.select_slider(
                    "Steady Level",
                    options=["Low", "Medium", "High"],
                    value="Medium",
                    help="The constant level of environmental uncertainty"
                )
                env_params["Steady Level"] = level
                
            elif env_pattern == "Increasing Complexity":
                growth = st.sidebar.select_slider(
                    "Complexity Growth",
                    options=["Slow", "Medium", "Fast"],
                    value="Medium",
                    help="How quickly environmental complexity increases"
                )
                env_params["Complexity Growth"] = growth
                
            elif env_pattern == "Learning Process":
                speed = st.sidebar.select_slider(
                    "Learning Rate",
                    options=["Slow", "Medium", "Fast"],
                    value="Medium",
                    help="How quickly environmental uncertainty decreases"
                )
                env_params["Learning Rate"] = speed
                
            elif env_pattern == "Cyclical Challenges":
                freq = st.sidebar.select_slider(
                    "Challenge Frequency",
                    options=["Low", "Medium", "High"],
                    value="Medium",
                    help="How often environmental challenges occur"
                )
                env_params["Challenge Frequency"] = freq
                
                intensity = st.sidebar.select_slider(
                    "Challenge Intensity",
                    options=["Mild", "Moderate", "Severe"],
                    value="Moderate",
                    help="How strong the environmental challenges are"
                )
                env_params["Challenge Intensity"] = intensity
                
            elif env_pattern == "Random Environment":
                level = st.sidebar.select_slider(
                    "Uncertainty Level",
                    options=["Low", "Medium", "High"],
                    value="Medium",
                    help="The average level of environmental uncertainty"
                )
                env_params["Uncertainty Level"] = level
                
                variability = st.sidebar.select_slider(
                    "Variability",
                    options=["Low", "Medium", "High"],
                    value="Medium",
                    help="How much the environmental uncertainty fluctuates"
                )
                env_params["Variability"] = variability
            
            basic_config["Environmental Parameters"] = env_params
            
        # Simulation time
        st.sidebar.subheader("3. Simulation Settings")
        sim_length = st.sidebar.select_slider(
            "Simulation Length",
            options=["Short (100)", "Medium (200)", "Long (300)", "Very Long (400)"],
            value="Medium (200)",
            help="How long to run the simulation"
        )
        
        # Parse the simulation length
        sim_time_map = {
            "Short (100)": 100,
            "Medium (200)": 200,
            "Long (300)": 300,
            "Very Long (400)": 400
        }
        sim_time = sim_time_map.get(sim_length, 200)
        time_steps = sim_time * 2  # 2 points per time unit
        
        basic_config["time_span"] = np.linspace(0, sim_time, time_steps)
    else:
        # When not customizing, we'll use the template's time span
        pass
    
    # Option to add description
    st.sidebar.subheader("4. Add Notes (Optional)")
    description = st.sidebar.text_area(
        "Experiment Notes",
        value=f"Based on {selected_template_name} template",
        height=80,
        help="Notes about this experiment run for future reference"
    )
    basic_config["description"] = description
    
    # Run experiment button
    if st.sidebar.button("Run Experiment"):
        with st.spinner("Running experiment..."):
            if custom_params:
                # If using custom parameters, translate from Basic to Expert
                expert_config = ParameterTranslator.translate_full_config(basic_config)
                
                # Configure experiment with translated parameters
                simulator.configure_experiment(
                    experiment_id=experiment_id,
                    **expert_config
                )
            else:
                # If using template as-is, apply it directly
                template.apply(simulator)
                
            # Run the experiment
            simulator.run_experiment(experiment_id)
            
            # Save the configuration for display
            if custom_params:
                st.session_state.current_basic_config = basic_config
            else:
                # For templates, we'll convert the expert parameters to basic for display
                template_params = template.parameters.copy()
                basic_params = {}
                
                for expert_name, expert_value in template_params.items():
                    if expert_name in ParameterTranslator.INVERSE_PARAMETER_NAME_MAP:
                        basic_name = ParameterTranslator.INVERSE_PARAMETER_NAME_MAP[expert_name]
                        basic_value = ParameterTranslator.expert_to_basic(expert_name, expert_value)
                        if basic_value:
                            basic_params[basic_name] = basic_value
                
                st.session_state.current_basic_config = {
                    "experiment_id": experiment_id,
                    "template_name": selected_template_name,
                    **basic_params,
                    "description": template.description
                }
            
            # Store the experiment ID for display
            st.session_state.current_experiment = experiment_id
            
            return basic_config
    
    return None

def render_experiment_results(simulator):
    """
    Render the enhanced experiment results visualization for Basic Mode.
    
    Args:
        simulator: The CoherenceSimulator instance
    """
    if 'current_experiment' not in st.session_state or not st.session_state.current_experiment:
        # Show instructions if no experiment has been run
        st.markdown("""
        ## Welcome to the Reflective Coherence Explorer
        
        This tool helps you explore how systems maintain internal consistency while adapting to changing environments.
        
        ### Getting Started:
        1. Choose an experiment template from the sidebar
        2. Optionally adjust parameters to customize the experiment
        3. Click "Run Experiment" to see the results
        
        Each template is designed to demonstrate a specific aspect of coherence dynamics.
        """)
        
        # Show template cards
        st.subheader("Available Templates")
        
        templates = get_all_templates()
        cols = st.columns(2)
        
        for i, (name, template) in enumerate(templates.items()):
            with cols[i % 2]:
                with st.expander(name, expanded=False):
                    st.markdown(template.description)
                    st.markdown("---")
                    
                    params = template.parameters
                    if "alpha" in params:
                        st.markdown(f"**System Growth Rate (α):** {params['alpha']}")
                    if "beta" in params:
                        st.markdown(f"**Entropy Influence (β):** {params['beta']}")
                    if "K" in params:
                        st.markdown(f"**Maximum Coherence (K):** {params['K']}")
        
        return
    
    # Get the current experiment ID and results
    experiment_id = st.session_state.current_experiment
    
    if experiment_id not in simulator.results:
        st.error(f"Results for experiment '{experiment_id}' not found")
        return
    
    results = simulator.results[experiment_id]
    
    # Get the current configuration for parameter display
    current_config = st.session_state.get('current_basic_config', {})
    
    # Get the actual scientific parameters for the plot
    experiment_params = simulator.experiments.get(experiment_id, {})
    
    # Create summary card
    summary = create_summary_card(results, experiment_id, experiment_params)
    
    # Display results header
    if "template_name" in current_config:
        st.header(f"{current_config['template_name']}: {experiment_id}")
    else:
        st.header(f"Experiment: {experiment_id}")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Basic Results", "Scientific Details"])
    
    with tab1:
        # Create two columns for the plot and summary
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create enhanced visualization
            fig = plot_with_interpretations(results, experiment_id, experiment_params)
            st.pyplot(fig)
        
        with col2:
            # Display summary card
            st.subheader("Results Summary")
            
            if summary["is_coherent"]:
                st.success("System ended in a COHERENT state")
            else:
                st.error("System ended in an INCOHERENT state")
            
            st.markdown("**Behavior Pattern:**")
            st.info(summary["behavior"])
            
            st.markdown("**Phase Transitions:**")
            st.info(f"{summary['num_transitions']} transitions")
            
            st.markdown("**Stability:**")
            stability_percentage = int(summary["stability"] * 100)
            st.progress(stability_percentage / 100)
            st.caption(f"{stability_percentage}% stable")
        
        # Display interpretation
        st.subheader("What This Means")
        st.markdown(summary["interpretation"])
        
        # If this was a template, show the template description
        if "template_name" in current_config:
            with st.expander("About This Template"):
                template = get_template(current_config["template_name"])
                if template:
                    st.markdown(template.description)
    
    with tab2:
        # Display more detailed scientific information
        st.subheader("Scientific Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Core Parameters:**")
            st.markdown(f"- Coherence Growth Rate (α): {experiment_params.get('alpha', 'N/A'):.3f}")
            st.markdown(f"- Entropy Influence (β): {experiment_params.get('beta', 'N/A'):.3f}")
            st.markdown(f"- Maximum Coherence (K): {experiment_params.get('K', 'N/A'):.2f}")
            st.markdown(f"- Initial Coherence: {experiment_params.get('initial_coherence', 'N/A'):.2f}")
        
        with col2:
            st.markdown("**Calculated Metrics:**")
            st.markdown(f"- Final Coherence: {summary['final_coherence']:.3f}")
            st.markdown(f"- Final Threshold: {summary['final_threshold']:.3f}")
            st.markdown(f"- Mean Entropy: {summary['mean_entropy']:.3f}")
            st.markdown(f"- Coherence Change: {summary['coherence_change']:.3f}")
        
        # Display raw data in a dataframe with a slider
        st.subheader("Raw Data")
        data = {
            "Time": results["time"],
            "Coherence": results["coherence"],
            "Entropy": results["entropy"],
            "Threshold": results["threshold"]
        }
        df = pd.DataFrame(data)
        
        # Show only a subset of rows with a slider
        row_count = len(df)
        if row_count > 20:
            show_rows = st.slider("Number of data points to display", 20, row_count, 100)
            # Select rows evenly spaced through the dataset
            indices = np.linspace(0, row_count-1, show_rows, dtype=int)
            df_display = df.iloc[indices]
        else:
            df_display = df
        
        st.dataframe(df_display)
        
        # Option to download the data
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f"{experiment_id}_data.csv",
            mime="text/csv",
        )

def render_basic_interface(simulator):
    """
    Render the complete Basic Mode interface.
    
    Args:
        simulator: The CoherenceSimulator instance
    """
    # Render the sidebar controls
    config = render_basic_sidebar(simulator)
    
    # Render the main content area with results
    render_experiment_results(simulator) 