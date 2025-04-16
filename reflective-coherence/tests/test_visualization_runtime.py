"""
Tests to verify that visualizations handle large datasets without runtime errors.

This file tests the visualization components with various dataset sizes and edge cases
to ensure they don't error out during runtime.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Mock streamlit for testing
sys.modules['streamlit'] = MagicMock()
mock_streamlit = sys.modules['streamlit']
mock_streamlit.cache_data = lambda func: func
mock_streamlit.cache_resource = lambda func: func

# Import visualization functions
from app.visualization.basic_visualizations import (
    downsample_time_series,
    detect_phase_transitions,
    identify_coherent_regions,
    detect_critical_slowing,
    plot_with_interpretations,
    create_summary_card
)

@pytest.fixture
def large_dataset():
    """Generate a large dataset for visualization testing."""
    # Create a large dataset with 10,000 points
    time = np.linspace(0, 1000, 10000)
    # Create a complex coherence signal with transitions
    coherence = 0.5 + 0.4 * np.sin(time * 0.01) + 0.1 * np.random.randn(10000)
    # Ensure coherence stays in valid range
    coherence = np.clip(coherence, 0, 1)
    # Create entropy values
    entropy = 0.3 + 0.2 * np.cos(time * 0.005) + 0.05 * np.random.randn(10000)
    # Create threshold
    threshold = 0.5 * np.ones_like(time)
    
    return time, coherence, entropy, threshold

@pytest.fixture
def edge_case_dataset():
    """Generate edge case datasets for testing visualization robustness."""
    # Very small dataset
    tiny = {
        'time': np.linspace(0, 10, 5),
        'coherence': np.array([0.1, 0.4, 0.6, 0.8, 0.7]),
        'entropy': np.array([0.3, 0.3, 0.2, 0.2, 0.3]),
        'threshold': 0.5
    }
    
    # Dataset with extreme values
    extreme = {
        'time': np.linspace(0, 100, 1000),
        'coherence': np.concatenate([np.zeros(50), np.ones(900), np.zeros(50)]),
        'entropy': np.concatenate([np.ones(100) * 0.9, np.zeros(800), np.ones(100) * 0.9]),
        'threshold': np.concatenate([np.ones(500) * 0.8, np.ones(500) * 0.2])
    }
    
    # Dataset with repeated values (potential division by zero issues)
    repeated = {
        'time': np.linspace(0, 100, 1000),
        'coherence': np.ones(1000) * 0.5,
        'entropy': np.ones(1000) * 0.5,
        'threshold': np.ones(1000) * 0.5
    }
    
    return {'tiny': tiny, 'extreme': extreme, 'repeated': repeated}

def test_downsample_large_dataset(large_dataset):
    """Test downsampling with large datasets."""
    time, coherence, entropy, threshold = large_dataset
    
    # Test downsampling to various sizes
    for target_size in [100, 500, 1000]:
        time_ds, coherence_ds = downsample_time_series(time, coherence, target_size)
        
        # Check that result is smaller than target size (plus potential buffer)
        assert len(time_ds) <= target_size + 1
        assert len(coherence_ds) <= target_size + 1
        
        # Test with preserve_features=True
        time_ds, coherence_ds = downsample_time_series(time, coherence, target_size, preserve_features=True)
        assert len(time_ds) <= target_size + len(time)//100  # Allow some extra points for features

def test_phase_transition_detection_performance(large_dataset):
    """Test phase transition detection with large datasets."""
    time, coherence, entropy, threshold = large_dataset
    
    # Time performance and check for errors with large dataset
    transitions = detect_phase_transitions(time, coherence, threshold)
    
    # Verify output is valid
    assert isinstance(transitions, list)
    
    # Ensure transitions are within the time range
    for t in transitions:
        assert time[0] <= t <= time[-1]

def test_coherent_regions_with_large_dataset(large_dataset):
    """Test coherent region identification with large datasets."""
    time, coherence, entropy, threshold = large_dataset
    
    # Test with large dataset
    regions = identify_coherent_regions(time, coherence, threshold)
    
    # Verify output structure
    assert isinstance(regions, list)
    for region in regions:
        assert len(region) == 2
        assert region[0] <= region[1]
        assert time[0] <= region[0] <= time[-1]
        assert time[0] <= region[1] <= time[-1]

def test_critical_slowing_with_large_dataset(large_dataset):
    """Test critical slowing detection with large datasets."""
    time, coherence, entropy, threshold = large_dataset
    
    # Test with large dataset
    critical_points = detect_critical_slowing(time, coherence)
    
    # Verify output
    assert isinstance(critical_points, list)
    for point in critical_points:
        assert time[0] <= point <= time[-1]

def test_plot_with_interpretations_large_dataset(large_dataset, monkeypatch):
    """Test plotting with large datasets."""
    time, coherence, entropy, threshold = large_dataset
    
    # Mock plt.show to avoid displaying the plot
    monkeypatch.setattr(plt, 'show', lambda: None)
    
    # Test with large dataset
    fig, summary = plot_with_interpretations(time, coherence, entropy, threshold, "Large Dataset Test")
    
    # Verify output
    assert isinstance(fig, plt.Figure)
    assert isinstance(summary, dict)
    assert "coherence_mean" in summary
    assert "phase_transitions" in summary

def test_plot_with_edge_cases(edge_case_dataset, monkeypatch):
    """Test plotting with edge case datasets."""
    # Mock plt.show to avoid displaying the plot
    monkeypatch.setattr(plt, 'show', lambda: None)
    
    for case_name, data in edge_case_dataset.items():
        # Test with edge case
        fig, summary = plot_with_interpretations(
            data['time'], data['coherence'], data['entropy'], data['threshold'], 
            f"Edge Case: {case_name}"
        )
        
        # Verify no errors
        assert isinstance(fig, plt.Figure)
        assert isinstance(summary, dict)

def test_summary_card_with_large_dataset(large_dataset):
    """Test summary card generation with large datasets."""
    time, coherence, entropy, threshold = large_dataset
    
    # Test with large dataset
    card = create_summary_card(time, coherence, entropy, threshold, "Large Dataset Test")
    
    # Verify output
    assert isinstance(card, dict)
    assert "metrics" in card
    assert "interpretation" in card
    assert isinstance(card["interpretation"], str)
    assert len(card["interpretation"]) > 0

def test_summary_card_with_edge_cases(edge_case_dataset):
    """Test summary card generation with edge case datasets."""
    for case_name, data in edge_case_dataset.items():
        # Test with edge case
        card = create_summary_card(
            data['time'], data['coherence'], data['entropy'], data['threshold'],
            f"Edge Case: {case_name}"
        )
        
        # Verify no errors
        assert isinstance(card, dict)
        assert "metrics" in card
        assert "interpretation" in card 