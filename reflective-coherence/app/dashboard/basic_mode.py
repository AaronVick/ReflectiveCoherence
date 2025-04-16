"""
Basic Mode Interface for Reflective Coherence Explorer.

This module provides a simplified user interface for the Reflective Coherence Explorer,
designed for users who want to understand coherence dynamics without diving into
the mathematical details. It leverages the parameter translation layer to convert
simplified inputs into precise scientific parameters.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import traceback
import logging
from time import time
import uuid

from reflective_coherence.simulator import CoherenceSimulator
from app.translation.parameter_mapping import ParameterTranslator
from app.templates.experiment_templates import get_all_templates, get_template_by_name
from app.visualization.basic_visualizations import plot_with_interpretations, create_summary_card, downsample_time_series

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("basic_mode")

# Cache expensive simulator initialization
@st.cache_resource
def get_simulator():
    """Initialize and return a CoherenceSimulator instance."""
    try:
        simulator = CoherenceSimulator()
        return simulator
    except Exception as e:
        logger.error(f"Error initializing simulator: {str(e)}")
        st.error("Failed to initialize simulator. Please refresh the page and try again.")
        return None

@st.cache_data
def prepare_experiment_data(time, coherence, entropy, threshold):
    """
    Prepare experiment data for visualization and storage.
    
    This function handles downsampling for storage efficiency while preserving
    important features of the data.
    
    Args:
        time: Array of time points
        coherence: Array of coherence values
        entropy: Array of entropy values
        threshold: Array of threshold values
        
    Returns:
        Dictionary with prepared data
    """
    try:
        # Ensure inputs are valid numpy arrays
        time = np.asarray(time)
        coherence = np.asarray(coherence)
        entropy = np.asarray(entropy)
        
        # Ensure non-negative values (physical constraint)
        coherence = np.maximum(coherence, 0)
        entropy = np.maximum(entropy, 0)
        
        # Downsample for storage efficiency if dataset is large
        if len(time) > 1000:
            time_ds, coherence_ds = downsample_time_series(time, coherence, 1000)
            _, entropy_ds = downsample_time_series(time, entropy, 1000)
            
            if isinstance(threshold, np.ndarray):
                _, threshold_ds = downsample_time_series(time, threshold, 1000)
            else:
                threshold_ds = threshold
        else:
            time_ds, coherence_ds, entropy_ds = time, coherence, entropy
            threshold_ds = threshold
        
        # Convert numpy arrays to lists for JSON serialization
        return {
            'time': time_ds.tolist() if isinstance(time_ds, np.ndarray) else time_ds,
            'coherence': coherence_ds.tolist() if isinstance(coherence_ds, np.ndarray) else coherence_ds,
            'entropy': entropy_ds.tolist() if isinstance(entropy_ds, np.ndarray) else entropy_ds,
            'threshold': threshold_ds.tolist() if isinstance(threshold_ds, np.ndarray) else threshold_ds
        }
    except Exception as e:
        logger.error(f"Error preparing experiment data: {str(e)}")
        # Return minimal valid data structure to prevent crashes
        return {
            'time': [0, 1],
            'coherence': [0.1, 0.1],
            'entropy': [0.1, 0.1],
            'threshold': 0.2,
            'error': str(e)
        }

def initialize_session_state():
    """
    Initialize session state variables for experiment data and error recovery.
    """
    if 'experiment_results' not in st.session_state:
        st.session_state.experiment_results = None
    
    if 'error_state' not in st.session_state:
        st.session_state.error_state = {
            'has_error': False,
            'error_message': '',
            'error_timestamp': None,
            'recovery_options': []
        }
    
    if 'experiment_history' not in st.session_state:
        st.session_state.experiment_history = []
    
    if 'last_successful_config' not in st.session_state:
        st.session_state.last_successful_config = None
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

def record_experiment_config(config):
    """
    Record experiment configuration for recovery purposes.
    
    Args:
        config: Dictionary containing experiment parameters
    """
    # Add timestamp and unique ID to the configuration
    config_record = config.copy()
    config_record['timestamp'] = time()
    config_record['config_id'] = str(uuid.uuid4())
    
    # Store in experiment history (limit to 10 most recent)
    st.session_state.experiment_history.append(config_record)
    if len(st.session_state.experiment_history) > 10:
        st.session_state.experiment_history = st.session_state.experiment_history[-10:]
    
    # Update last successful config if operation succeeds
    st.session_state.last_successful_config = config_record

def show_basic_mode_interface():
    """Display the Basic Mode interface with simplified controls and enhanced visualizations."""
    st.title("Reflective Coherence Explorer - Basic Mode")
    
    # Initialize session state
    initialize_session_state()
    
    # Display error banner if there's an active error
    if st.session_state.error_state['has_error']:
        with st.expander("⚠️ Error Information (click to expand)", expanded=True):
            st.error(st.session_state.error_state['error_message'])
            
            # Offer recovery options
            st.subheader("Recovery Options")
            if st.button("Clear Error and Continue"):
                st.session_state.error_state['has_error'] = False
                st.experimental_rerun()
            
            if st.session_state.last_successful_config and st.button("Restore Last Working Configuration"):
                # Restore the last known good configuration
                st.session_state.error_state['has_error'] = False
                # Configuration will be restored below
                st.experimental_rerun()
    
    st.markdown("""
        This simplified interface helps you explore coherence dynamics with intuitive controls.
        Choose a pre-defined experiment template or customize your own experiment with the sliders.
    """)
    
    # Layout with two columns
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Experiment Setup")
        
        # Get cached simulator with error handling
        simulator = get_simulator()
        if simulator is None:
            st.error("Failed to initialize simulator. Try refreshing the page.")
            return
        
        # Template selection
        try:
            templates = get_all_templates()
            template_names = [template.name for template in templates]
            
            use_template = st.checkbox("Use a pre-defined experiment template", value=True)
            
            if use_template:
                selected_template_name = st.selectbox(
                    "Select an experiment template",
                    options=template_names,
                    index=0
                )
                
                # Get the selected template and display its description
                template = get_template_by_name(selected_template_name)
                st.markdown(f"**Description:** {template.description}")
                
                # Apply template button
                if st.button("Apply Template"):
                    try:
                        # Record configuration for recovery
                        record_experiment_config({
                            'type': 'template',
                            'template_name': selected_template_name
                        })
                        
                        # Apply template to simulator
                        simulator = get_simulator()
                        template.apply_to_simulator(simulator)
                        
                        # Run simulation and store optimized results
                        run_experiment(simulator)
                    except Exception as e:
                        handle_error(e, "Error applying template")
        except Exception as e:
            handle_error(e, "Error loading templates")
            st.error("Failed to load experiment templates. Using custom experiment mode instead.")
            use_template = False
        
        # Divider
        st.markdown("---")
        
        # Custom experiment controls
        st.subheader("Custom Experiment")
        
        try:
            # Simplified sliders
            growth_speed = st.slider(
                "Growth Speed", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5,
                help="How quickly coherence develops (higher = faster growth)"
            )
            
            uncertainty_impact = st.slider(
                "Uncertainty Impact", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.3,
                help="How strongly uncertainty affects coherence (higher = stronger impact)"
            )
            
            system_capacity = st.slider(
                "System Capacity", 
                min_value=0.5, 
                max_value=2.0, 
                value=1.0,
                help="Maximum potential coherence level (higher = more capacity)"
            )
            
            starting_stability = st.slider(
                "Starting Stability", 
                min_value=0.0, 
                max_value=0.5, 
                value=0.1,
                help="Initial coherence level (higher = more stable start)"
            )
            
            # Entropy type selection (simplified)
            entropy_options = ["Constant", "Increasing", "Oscillating", "Random"]
            entropy_type = st.selectbox(
                "Entropy Pattern",
                options=entropy_options,
                index=0,
                help="Pattern of uncertainty over time"
            )
            
            # Simulation duration
            time_span = st.slider(
                "Simulation Duration", 
                min_value=20, 
                max_value=200, 
                value=100,
                help="How many time steps to simulate"
            )
            
            # Run custom experiment button
            if st.button("Run Custom Experiment"):
                try:
                    # Store configuration for recovery
                    custom_config = {
                        'type': 'custom',
                        'growth_speed': growth_speed,
                        'uncertainty_impact': uncertainty_impact,
                        'system_capacity': system_capacity,
                        'starting_stability': starting_stability,
                        'entropy_type': entropy_type,
                        'time_span': time_span
                    }
                    record_experiment_config(custom_config)
                    
                    # Translate parameters from Basic to Expert mode
                    basic_params = {
                        "growth_speed": growth_speed,
                        "uncertainty_impact": uncertainty_impact,
                        "system_capacity": system_capacity,
                        "starting_stability": starting_stability,
                        "entropy_type": entropy_type,
                        "time_span": time_span
                    }
                    
                    # Translate to expert parameters
                    expert_params = ParameterTranslator.basic_to_expert(basic_params)
                    
                    # Set up simulator with translated parameters
                    simulator = get_simulator()
                    simulator.configure_experiment(
                        experiment_id=f"custom_{int(time())}",
                        alpha=expert_params["alpha"],
                        beta=expert_params["beta"],
                        K=expert_params["K"],
                        initial_coherence=expert_params["initial_coherence"],
                        entropy_fn=expert_params["entropy_fn"],
                        time_span=np.linspace(0, time_span, min(time_span * 5, 1000))  # Cap for performance
                    )
                    
                    # Run simulation with storage optimization
                    run_experiment(simulator)
                except Exception as e:
                    handle_error(e, "Error running custom experiment")
        except Exception as e:
            handle_error(e, "Error in experiment setup")
            st.error("Failed to set up experiment controls. Please refresh the page.")
        
        # Add auto-recovery information
        st.markdown("---")
        with st.expander("Experiment History", expanded=False):
            if st.session_state.experiment_history:
                for i, config in enumerate(reversed(st.session_state.experiment_history)):
                    if st.button(f"Restore {config['type'].title()} Experiment from {int(time() - config['timestamp'])} seconds ago", key=f"restore_{i}"):
                        restore_experiment_config(config)
                        st.experimental_rerun()
            else:
                st.info("No experiment history available yet.")
    
    with col2:
        st.subheader("Experiment Results")
        
        if st.session_state.experiment_results is not None:
            try:
                display_basic_results()
            except Exception as e:
                handle_error(e, "Error displaying results")
                st.error("Failed to display results. Try running the experiment again.")
                # Show a placeholder for the results
                st.markdown("### Results unavailable")
                st.markdown("The visualization could not be displayed due to an error.")
        else:
            st.info("Run an experiment to see results here.")

def run_experiment(simulator, time_span=None):
    """
    Run an experiment with the configured simulator and store the results.
    
    Args:
        simulator: Configured CoherenceSimulator instance
        time_span: Optional override for time span
    """
    try:
        # Track the start time for performance logging
        start_time = time()
        
        # Get experiment IDs from simulator
        experiment_ids = list(simulator.experiments.keys())
        
        if not experiment_ids:
            st.error("No experiment configured. Please configure an experiment first.")
            return
        
        # Use the most recent experiment
        experiment_id = experiment_ids[-1]
        
        # Override time span if specified
        if time_span is not None:
            simulator.experiments[experiment_id]['time_span'] = np.linspace(0, time_span, min(time_span * 5, 1000))
        
        # Run the experiment
        simulator.run_experiment(experiment_id)
        
        # Get the results
        results = simulator.results[experiment_id]
        
        # Calculate threshold if not already in results
        if 'threshold' not in results:
            threshold = simulator.get_experiment_summary(experiment_id)["results"]["threshold"]
            threshold_array = np.ones_like(results['time']) * threshold
            results['threshold'] = threshold_array
        
        # Prepare data for storage in session state (with optimization)
        prepared_data = prepare_experiment_data(
            results['time'],
            results['coherence'],
            results['entropy'],
            results['threshold']
        )
        
        # Store experiment parameters
        params = simulator.experiments[experiment_id].copy()
        
        # Remove non-serializable items
        if 'entropy_fn' in params:
            params['entropy_fn'] = str(params['entropy_fn'])
        if 'time_span' in params:
            params['time_span'] = {
                'start': float(params['time_span'][0]),
                'end': float(params['time_span'][-1]),
                'points': len(params['time_span'])
            }
        
        # Store results in session state
        st.session_state.experiment_results = {
            'data': prepared_data,
            'parameters': params,
            'experiment_id': experiment_id,
            'execution_time': time() - start_time
        }
        
        # Log successful execution
        logger.info(f"Experiment {experiment_id} executed successfully in {time() - start_time:.2f} seconds")
        
    except Exception as e:
        handle_error(e, "Error running experiment")
        st.error(f"Failed to run experiment: {str(e)}")
        
        # Create minimal emergency results if none exist
        if 'experiment_results' not in st.session_state or st.session_state.experiment_results is None:
            st.session_state.experiment_results = {
                'data': {
                    'time': list(range(10)),
                    'coherence': [0.1] * 10,
                    'entropy': [0.2] * 10,
                    'threshold': 0.3
                },
                'parameters': {'error': True},
                'experiment_id': 'error_recovery',
                'execution_time': 0
            }

def handle_error(exception, context="Error"):
    """
    Handle errors gracefully with logging and recovery options.
    
    Args:
        exception: The exception that occurred
        context: Context description for the error
    """
    # Get full stack trace
    stack_trace = traceback.format_exc()
    
    # Log the error
    logger.error(f"{context}: {str(exception)}\n{stack_trace}")
    
    # Update error state
    st.session_state.error_state = {
        'has_error': True,
        'error_message': f"{context}: {str(exception)}",
        'error_timestamp': time(),
        'recovery_options': ['clear', 'restore_last']
    }

def restore_experiment_config(config):
    """
    Restore a previously used experiment configuration.
    
    Args:
        config: The configuration to restore
    """
    try:
        simulator = get_simulator()
        
        if config['type'] == 'template':
            template = get_template_by_name(config['template_name'])
            template.apply_to_simulator(simulator)
            run_experiment(simulator)
        elif config['type'] == 'custom':
            # Translate parameters from Basic to Expert mode
            basic_params = {
                "growth_speed": config['growth_speed'],
                "uncertainty_impact": config['uncertainty_impact'],
                "system_capacity": config['system_capacity'],
                "starting_stability": config['starting_stability'],
                "entropy_type": config['entropy_type'],
                "time_span": config['time_span']
            }
            
            # Translate to expert parameters
            expert_params = ParameterTranslator.basic_to_expert(basic_params)
            
            # Configure simulator
            simulator.configure_experiment(
                experiment_id=f"restored_{int(time())}",
                alpha=expert_params["alpha"],
                beta=expert_params["beta"],
                K=expert_params["K"],
                initial_coherence=expert_params["initial_coherence"],
                entropy_fn=expert_params["entropy_fn"],
                time_span=np.linspace(0, config['time_span'], min(config['time_span'] * 5, 1000))
            )
            
            # Run simulation
            run_experiment(simulator)
        
        # Clear error state upon successful restoration
        st.session_state.error_state['has_error'] = False
        
    except Exception as e:
        logger.error(f"Error restoring configuration: {str(e)}")
        st.error(f"Failed to restore configuration: {str(e)}")

@st.cache_data
def generate_results_visualization(data, parameters=None):
    """
    Generate visualization from experiment results.
    
    Args:
        data: Dictionary with time, coherence, entropy, and threshold data
        parameters: Optional experiment parameters
        
    Returns:
        Tuple of (figure, summary_data)
    """
    try:
        # Convert lists back to numpy arrays if needed
        time = np.array(data['time'])
        coherence = np.array(data['coherence'])
        entropy = np.array(data['entropy'])
        threshold = data['threshold']
        
        if isinstance(threshold, list):
            threshold = np.array(threshold)
        
        # Generate enhanced visualization
        fig, summary = plot_with_interpretations(
            time, coherence, entropy, threshold,
            experiment_name=parameters.get('description', 'Experiment') if parameters else 'Experiment'
        )
        
        return fig, summary
        
    except Exception as e:
        # Create error figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error generating visualization: {str(e)}",
                ha='center', va='center', fontsize=12)
        ax.set_axis_off()
        
        # Return error information
        return fig, {"error": str(e)}

def display_basic_results():
    """Display the results of the experiment with interpretations."""
    if st.session_state.experiment_results is None:
        st.info("Run an experiment to see results.")
        return
    
    try:
        # Get data and parameters
        data = st.session_state.experiment_results['data']
        parameters = st.session_state.experiment_results.get('parameters', {})
        execution_time = st.session_state.experiment_results.get('execution_time', 0)
        
        # Check for error in data
        if 'error' in data:
            st.error(f"Error in data: {data['error']}")
            return
        
        # Generate visualization (cached)
        fig, summary = generate_results_visualization(data, parameters)
        
        # Check for error in visualization
        if isinstance(summary, dict) and 'error' in summary:
            st.error(f"Error generating visualization: {summary['error']}")
            
        # Create summary card with experiment interpretation
        try:
            time_array = np.array(data['time'])
            coherence_array = np.array(data['coherence'])
            entropy_array = np.array(data['entropy'])
            threshold_value = data['threshold']
            
            card = create_summary_card(
                time_array, coherence_array, entropy_array, threshold_value,
                experiment_name=parameters.get('description', 'Experiment')
            )
            
            # Display summary card
            with st.expander("Experiment Summary", expanded=True):
                st.markdown(f"### {card['title']}")
                
                # Display metrics
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Average Coherence", f"{card['metrics']['average_coherence']:.2f}")
                    st.metric("Phase Transitions", card['metrics']['phase_transitions'])
                with metrics_col2:
                    st.metric("Maximum Coherence", f"{card['metrics']['max_coherence']:.2f}")
                    st.metric("Time in Coherent State", f"{card['metrics']['coherent_time']:.1f}%")
                
                # Display interpretation
                st.markdown("#### Interpretation")
                st.markdown(card['interpretation'])
                
                # Display performance information
                st.caption(f"Experiment executed in {execution_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error creating summary card: {str(e)}")
            st.warning("Could not generate detailed summary. See visualization for results.")
        
        # Display the visualization
        st.pyplot(fig)
        
        # Display export and analysis options
        with st.expander("Export and Advanced Analysis", expanded=False):
            # Create a DataFrame for download
            df = pd.DataFrame({
                'Time': data['time'],
                'Coherence': data['coherence'],
                'Entropy': data['entropy']
            })
            
            # Convert to CSV for download
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name=f"coherence_experiment_{st.session_state.experiment_results['experiment_id']}.csv",
                mime="text/csv",
            )
            
            # Option to view raw data
            if st.checkbox("View Raw Data"):
                st.dataframe(df)
    
    except Exception as e:
        handle_error(e, "Error displaying results")
        st.error("Failed to display results. Please try running the experiment again.") 