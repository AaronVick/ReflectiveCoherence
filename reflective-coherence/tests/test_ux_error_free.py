import pytest
import sys
import os
from pathlib import Path
import numpy as np
from unittest.mock import MagicMock, patch

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Create mocks for external libraries
sys.modules['streamlit'] = MagicMock()
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()
sys.modules['plotly.express'] = MagicMock()

# Set up streamlit mock attributes
mock_streamlit = sys.modules['streamlit']
mock_streamlit.cache_data = lambda func: func
mock_streamlit.cache_resource = lambda func: func
mock_streamlit.session_state = {}
mock_streamlit.columns = lambda *args, **kwargs: [MagicMock(), MagicMock()]
mock_streamlit.expander = lambda *args, **kwargs: MagicMock()

# Tests to verify UX files don't error out

@pytest.fixture
def mock_simulator():
    """Mock simulator to test UX components."""
    with patch('core.simulators.coherence_simulator.CoherenceSimulator') as mock_sim:
        simulator = mock_sim.return_value
        
        # Set up mock experiment data
        simulator.experiments = {"test_exp": {
            "alpha": 0.5,
            "beta": 0.2,
            "K": 1.0,
            "initial_coherence": 0.1,
            "entropy_fn": lambda t: 0.3,
            "time_span": np.linspace(0, 100, 500),
            "description": "Test Experiment"
        }}
        
        # Set up mock results
        simulator.results = {"test_exp": {
            "time": np.linspace(0, 100, 500),
            "coherence": np.linspace(0.1, 0.8, 500),
            "entropy": np.ones(500) * 0.3,
        }}
        
        # Mock get_experiment_summary
        simulator.get_experiment_summary.return_value = {
            "results": {
                "threshold": 0.5,
                "coherence_mean": 0.5,
                "coherence_max": 0.8
            }
        }
        
        # Mock run_experiment to return results
        simulator.run_experiment.return_value = simulator.results["test_exp"]
        
        yield simulator

def test_basic_mode_initialization():
    """Test that the basic_mode module can be imported without errors."""
    from app.dashboard import basic_mode
    
    # Verify critical functions exist
    assert hasattr(basic_mode, 'get_simulator')
    assert hasattr(basic_mode, 'show_basic_mode_interface')
    assert hasattr(basic_mode, 'run_experiment')
    assert hasattr(basic_mode, 'display_basic_results')

def test_app_initialization():
    """Test that the main app module can be imported without errors."""
    from app.dashboard import app
    
    # Verify critical functions exist
    assert hasattr(app, 'main')

@patch('app.dashboard.basic_mode.get_simulator')
def test_run_experiment_no_errors(mock_get_simulator, mock_simulator):
    """Test that run_experiment executes without errors."""
    from app.dashboard import basic_mode
    
    # Setup mock
    mock_get_simulator.return_value = mock_simulator
    
    # Initialize session state
    basic_mode.initialize_session_state()
    
    # Call the function
    basic_mode.run_experiment(mock_simulator)
    
    # Verify simulator was called
    mock_simulator.run_experiment.assert_called()

@patch('app.dashboard.basic_mode.get_simulator')
@patch('app.dashboard.basic_mode.generate_results_visualization')
def test_display_results_no_errors(mock_generate_viz, mock_get_simulator, mock_simulator):
    """Test that display_basic_results executes without errors."""
    from app.dashboard import basic_mode
    
    # Setup mocks
    mock_get_simulator.return_value = mock_simulator
    mock_generate_viz.return_value = (MagicMock(), {
        "coherence_mean": 0.5,
        "coherence_final": 0.8,
        "entropy_mean": 0.3,
        "threshold": 0.5,
        "phase_transitions": 1,
        "transition_times": [50],
        "coherent_regions": 1,
        "critical_points": 0
    })
    
    # Initialize session state and set experiment results
    basic_mode.initialize_session_state()
    mock_streamlit.session_state.experiment_results = {
        'data': {
            'time': list(range(100)),
            'coherence': [0.1 + i*0.007 for i in range(100)],
            'entropy': [0.3] * 100,
            'threshold': 0.5
        },
        'parameters': {'description': 'Test'},
        'experiment_id': 'test_exp',
        'execution_time': 0.5
    }
    
    # Call the function
    basic_mode.display_basic_results()
    
    # Verify visualization was generated
    mock_generate_viz.assert_called()

def test_parameter_translation_no_errors():
    """Test that parameter translation functions work without errors."""
    from app.translation.parameter_mapping import ParameterTranslator
    
    # Test basic to expert translation
    basic_params = {
        "growth_speed": "Medium",
        "uncertainty_impact": "Medium",
        "system_capacity": "Medium",
        "starting_stability": "Medium",
        "entropy_type": "Constant"
    }
    
    # Call the translation function
    expert_params = {}
    for param_name, param_value in basic_params.items():
        if param_name in ParameterTranslator.PARAMETER_NAME_MAP:
            expert_name = ParameterTranslator.PARAMETER_NAME_MAP[param_name]
            expert_params[expert_name] = ParameterTranslator.basic_to_expert(param_name, param_value)
    
    # Verify we got valid outputs
    assert 'alpha' in expert_params
    assert 'beta' in expert_params
    assert 'K' in expert_params
    assert 'initial_coherence' in expert_params

def test_visualization_components_no_errors():
    """Test that visualization components can be instantiated without errors."""
    from app.visualization.basic_visualizations import (
        downsample_time_series,
        detect_phase_transitions,
        identify_coherent_regions,
        detect_critical_slowing
    )
    
    # Create test data
    time = np.linspace(0, 100, 1000)
    coherence = np.sin(time * 0.1) * 0.4 + 0.5
    entropy = np.ones_like(time) * 0.3
    threshold = 0.5
    
    # Test downsampling
    time_ds, coherence_ds = downsample_time_series(time, coherence, 100)
    assert len(time_ds) <= 101
    
    # Test phase transition detection
    transitions = detect_phase_transitions(time, coherence, threshold)
    assert isinstance(transitions, list)
    
    # Test coherent region identification
    regions = identify_coherent_regions(time, coherence, threshold)
    assert isinstance(regions, list)
    
    # Test critical slowing detection
    critical_points = detect_critical_slowing(time, coherence)
    assert isinstance(critical_points, list)

# We're testing if the files can be imported without raising errors
# Even if we can't fully test functionality without dependencies
@patch('core.simulators.coherence_simulator.CoherenceSimulator')
def test_basic_mode_module_imports(mock_class):
    """Test that the basic_mode module can be imported without errors."""
    sys.modules['reflective_coherence'] = MagicMock()
    sys.modules['reflective_coherence.simulator'] = MagicMock()
    sys.modules['reflective_coherence.simulator.CoherenceSimulator'] = mock_class
    
    try:
        from app.dashboard import basic_mode
        assert True, "Module imported without errors"
    except ImportError:
        pytest.fail("basic_mode module failed to import")

@patch('core.simulators.coherence_simulator.CoherenceSimulator')
def test_parameter_translation_module_imports(mock_class):
    """Test that parameter mapping can be imported without errors."""
    try:
        from app.translation import parameter_mapping
        assert True, "Module imported without errors"
    except ImportError:
        pytest.fail("parameter_mapping module failed to import")
        
    # Basic check of module structure
    from app.translation.parameter_mapping import ParameterTranslator
    assert hasattr(ParameterTranslator, 'GROWTH_SPEED_MAP')
    assert hasattr(ParameterTranslator, 'UNCERTAINTY_IMPACT_MAP')
    assert hasattr(ParameterTranslator, 'basic_to_expert') 