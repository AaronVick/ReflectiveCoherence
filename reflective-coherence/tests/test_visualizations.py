import pytest
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Mock streamlit for testing purposes
import sys
from unittest.mock import MagicMock

# Create a mock for streamlit
mock_streamlit = MagicMock()
mock_streamlit.cache_data = lambda func: func  # Mock decorator to pass through the function
mock_streamlit.cache_resource = lambda func: func  # Mock decorator to pass through the function
sys.modules['streamlit'] = mock_streamlit

# Import simulator and visualization modules with correct paths
from core.simulators.coherence_simulator import CoherenceSimulator
from app.visualization.basic_visualizations import (
    detect_phase_transitions,
    identify_coherent_regions,
    detect_critical_slowing,
    plot_with_interpretations,
    create_summary_card,
    downsample_time_series  # Import the new function
)

@pytest.fixture
def sample_simulation_data():
    """Fixture to generate sample simulation data for testing visualizations."""
    # Create a simulator with parameters that will produce clear phase transitions
    simulator = CoherenceSimulator()
    
    # Configure with parameters that will produce clear phase transitions
    simulator.configure_experiment(
        experiment_id="test_visual",
        alpha=0.8,
        beta=0.2,
        K=1.0,
        initial_coherence=0.1,
        entropy_fn=lambda t: 0.3,  # Constant entropy
        time_span=np.linspace(0, 100, 500)
    )
    
    # Run the experiment
    simulator.run_experiment("test_visual")
    
    # Return the simulation results
    return {
        'time': simulator.results["test_visual"]["time"],
        'coherence': simulator.results["test_visual"]["coherence"],
        'entropy': simulator.results["test_visual"]["entropy"],
        'threshold': [simulator.get_experiment_summary("test_visual")["results"]["threshold"]] * len(simulator.results["test_visual"]["time"])
    }

# Add test for the new downsample_time_series function
def test_downsample_time_series():
    """Test that the downsampling function correctly reduces dataset size."""
    # Create test data
    time = np.linspace(0, 100, 1000)
    values = np.sin(time * 0.1)
    
    # Test downsampling to 100 points
    time_ds, values_ds = downsample_time_series(time, values, 100)
    
    # Check that output has correct length
    assert len(time_ds) <= 101  # Allow for including last point
    assert len(values_ds) <= 101
    
    # Check that we retain the first and last points
    assert time_ds[0] == time[0]
    assert abs(time_ds[-1] - time[-1]) < 1e-10
    
    # Check that values are a subset of the original data
    # Note: We can't directly check if the downsampled values are in the original
    # array due to floating point precision, so we check if they're close
    for t in time_ds:
        assert any(abs(t - orig_t) < 1e-10 for orig_t in time)

def test_phase_transition_detection(sample_simulation_data):
    """Test that phase transitions are correctly detected in simulation data."""
    transitions = detect_phase_transitions(
        sample_simulation_data['time'],
        sample_simulation_data['coherence'],
        sample_simulation_data['threshold']
    )
    
    # There should be at least one phase transition
    assert len(transitions) > 0
    
    # Transitions should be within the time range
    for t in transitions:
        assert 0 <= t <= 100
    
    # At transition points, coherence should be close to threshold
    for t in transitions:
        idx = np.argmin(np.abs(sample_simulation_data['time'] - t))
        coh = sample_simulation_data['coherence'][idx]
        thresh = sample_simulation_data['threshold'][idx]
        assert abs(coh - thresh) < 0.1, f"At t={t}, coherence={coh} should be close to threshold={thresh}"

def test_coherent_regions_identification(sample_simulation_data):
    """Test that coherent regions are correctly identified in simulation data."""
    regions = identify_coherent_regions(
        sample_simulation_data['time'],
        sample_simulation_data['coherence'],
        sample_simulation_data['threshold']
    )
    
    # There should be at least one coherent region
    assert len(regions) > 0
    
    # Each region should be a tuple of (start, end)
    for region in regions:
        assert len(region) == 2
        assert region[0] < region[1]
        
    # Coherence should be above threshold within coherent regions
    for start, end in regions:
        start_idx = np.argmin(np.abs(sample_simulation_data['time'] - start))
        end_idx = np.argmin(np.abs(sample_simulation_data['time'] - end))
        
        for i in range(start_idx, end_idx + 1):
            assert sample_simulation_data['coherence'][i] >= sample_simulation_data['threshold'][i], \
                f"At t={sample_simulation_data['time'][i]}, coherence should be above threshold"

def test_critical_slowing_detection(sample_simulation_data):
    """Test that critical slowing is correctly detected in simulation data."""
    slowing_points = detect_critical_slowing(
        sample_simulation_data['time'],
        sample_simulation_data['coherence']
    )
    
    # Type check
    assert isinstance(slowing_points, list)
    
    # If slowing points are detected, they should be within the time range
    for t in slowing_points:
        assert 0 <= t <= 100

def test_plot_with_interpretations(sample_simulation_data, monkeypatch):
    """Test that the plot with interpretations function executes without error."""
    # Mock plt.show to avoid displaying the plot during tests
    monkeypatch.setattr(plt, 'show', lambda: None)
    
    # Call the plotting function and verify it executes without error
    fig, summary = plot_with_interpretations(
        sample_simulation_data['time'],
        sample_simulation_data['coherence'],
        sample_simulation_data['entropy'],
        sample_simulation_data['threshold'],
        "Test Experiment"
    )
    
    # Verify that the figure was created
    assert isinstance(fig, plt.Figure)
    
    # Verify that a summary was returned
    assert isinstance(summary, dict)
    assert 'coherence_mean' in summary
    assert 'phase_transitions' in summary
    assert 'coherent_regions' in summary

def test_summary_card_creation(sample_simulation_data):
    """Test that the summary card is correctly created with interpretation."""
    summary = create_summary_card(
        sample_simulation_data['time'],
        sample_simulation_data['coherence'],
        sample_simulation_data['entropy'],
        sample_simulation_data['threshold'],
        "Test Experiment"
    )
    
    # Verify the summary structure
    assert isinstance(summary, dict)
    assert 'title' in summary
    assert 'metrics' in summary
    assert 'interpretation' in summary
    
    # Verify metrics
    assert 'average_coherence' in summary['metrics']
    assert 'max_coherence' in summary['metrics']
    assert 'phase_transitions' in summary['metrics']
    assert 'coherent_time' in summary['metrics']
    
    # Verify interpretation is a non-empty string
    assert isinstance(summary['interpretation'], str)
    assert len(summary['interpretation']) > 0 