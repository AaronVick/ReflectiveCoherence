"""
Pytest configuration and shared fixtures for the Reflective Coherence Explorer test suite.
"""

import pytest
import sys
import os
import tempfile
import shutil
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project modules
from core.models.coherence_model import CoherenceModel
from core.simulators.coherence_simulator import CoherenceSimulator

@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root path."""
    return project_root

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after the test
    shutil.rmtree(temp_dir)

@pytest.fixture
def standard_time_span():
    """Return a standard time span for simulations."""
    return np.linspace(0, 100, 500)

@pytest.fixture
def basic_model():
    """Return a basic coherence model with default parameters."""
    return CoherenceModel(
        alpha=0.1,
        K=1.0,
        beta=0.2,
        initial_coherence=0.5,
        entropy_fn=lambda t: 0.3  # Constant entropy
    )

@pytest.fixture
def oscillating_entropy_model():
    """Return a model with oscillating entropy."""
    return CoherenceModel(
        alpha=0.1,
        K=1.0,
        beta=0.2,
        initial_coherence=0.5,
        entropy_fn=lambda t: 0.3 + 0.1 * np.sin(0.1 * t)  # Oscillating entropy
    )

@pytest.fixture
def basic_simulator(temp_data_dir):
    """Return a basic simulator with a temporary data directory."""
    simulator = CoherenceSimulator(data_dir=temp_data_dir)
    
    # Configure a standard experiment
    simulator.configure_experiment(
        experiment_id="standard",
        alpha=0.1,
        K=1.0,
        beta=0.2,
        initial_coherence=0.5,
        entropy_fn=lambda t: 0.3,
        description="Standard test experiment"
    )
    
    return simulator

# Configure pytest behavior
def pytest_configure(config):
    """Configure pytest environment."""
    # Register custom markers
    config.addinivalue_line("markers", "model: Tests for the core coherence model")
    config.addinivalue_line("markers", "simulator: Tests for the coherence simulator")
    config.addinivalue_line("markers", "math: Tests for mathematical accuracy")
    config.addinivalue_line("markers", "integration: Integration tests")

def pytest_collection_modifyitems(items):
    """Apply markers to tests based on filenames."""
    for item in items:
        if "test_coherence_model" in item.nodeid:
            item.add_marker(pytest.mark.model)
        if "test_coherence_simulator" in item.nodeid:
            item.add_marker(pytest.mark.simulator)
        if "test_mathematical_accuracy" in item.nodeid:
            item.add_marker(pytest.mark.math)
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration) 