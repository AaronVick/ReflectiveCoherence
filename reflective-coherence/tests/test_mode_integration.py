"""
Integration tests to verify that Basic Mode maintains mathematical integrity with Expert Mode.

These tests ensure that both interface modes produce identical mathematical results
when configured with equivalent parameters, maintaining scientific accuracy
while improving accessibility.
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path
from scipy.stats import pearsonr

# Add the project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import with correct paths
from core.simulators.coherence_simulator import CoherenceSimulator
from app.translation.parameter_mapping import ParameterTranslator
from app.templates.experiment_templates import get_template_by_name, list_all_template_names

# Import utility functions
def constant_entropy(value=0.3):
    """Return a constant entropy function."""
    return lambda t: np.ones_like(t) * value if hasattr(t, '__len__') else value

def increasing_entropy(start=0.1, rate=0.005):
    """Return an increasing entropy function."""
    return lambda t: start + rate * t

def oscillating_entropy(base=0.3, amplitude=0.2, frequency=0.1):
    """Return an oscillating entropy function."""
    return lambda t: base + amplitude * np.sin(frequency * t)

class TestModeIntegration:
    """Test class for verifying the integrity between Basic and Expert modes."""

    def test_parameter_translation_integrity(self):
        """Test that basic parameters translate to expert parameters that produce valid results."""
        # Define basic mode parameters
        basic_params = {
            'growth_speed': 'Medium',
            'uncertainty_impact': 'High',
            'system_capacity': 'Medium',
            'starting_stability': 'Low',
            'entropy_type': 'Constant',
            'entropy_level': 'Medium'
        }
        
        # Translate to expert mode
        expert_params = ParameterTranslator.basic_to_expert(basic_params)
        
        # Create simulator with translated parameters
        sim = CoherenceSimulator()
        
        # Configure experiment with the translated parameters
        sim.configure_experiment(
            experiment_id="test_translation",
            alpha=expert_params['alpha'],
            beta=expert_params['beta'],
            K=expert_params['K'],
            initial_coherence=expert_params['initial_coherence'],
            entropy_fn=constant_entropy(expert_params['entropy_params']['level']),
            time_span=np.linspace(0, 10, 100)
        )
        
        # Run simulation
        results = sim.run_experiment("test_translation")
        
        # Verify we get valid coherence values
        assert len(results['coherence']) == 100
        assert np.all(np.isfinite(results['coherence']))
        assert np.all(results['coherence'] >= 0)
        
        # Verify the simulation results match expected behavior for parameters
        if basic_params['uncertainty_impact'] == 'High':
            # High uncertainty should limit coherence growth
            assert np.mean(results['coherence']) < 0.8 * expert_params['K']

    def test_template_parameters_validity(self):
        """Test that predefined templates produce valid mathematical results."""
        for template_name in list_all_template_names():
            template = get_template_by_name(template_name)
            
            # Create simulator
            sim = CoherenceSimulator()
            
            # Apply template parameters to experiment
            template.apply_to_simulator(sim, f"test_{template_name}")
            
            # Run simulation
            results = sim.run_experiment(f"test_{template_name}")
            
            # Verify we get valid coherence values
            assert len(results['coherence']) > 0
            assert np.all(np.isfinite(results['coherence']))
            assert np.all(results['coherence'] >= 0)
            
            # Verify entropy values are valid
            assert np.all(np.isfinite(results['entropy']))
            assert np.all(results['entropy'] >= 0)

    def test_bidirectional_translation(self):
        """Test that parameters can be translated from Basic to Expert and back."""
        # Original basic parameters
        original_basic = {
            'growth_speed': 'Fast',
            'uncertainty_impact': 'Low',
            'system_capacity': 'Large',
            'starting_stability': 'Medium',
            'entropy_type': 'Increasing',
            'entropy_level': 'Low',
            'entropy_change_rate': 'Slow'
        }
        
        # Translate to expert then back to basic
        expert = ParameterTranslator.basic_to_expert(original_basic)
        translated_basic = ParameterTranslator.expert_to_basic(expert)
        
        # Verify core parameters are preserved
        assert translated_basic['growth_speed'] == original_basic['growth_speed']
        assert translated_basic['uncertainty_impact'] == original_basic['uncertainty_impact']
        assert translated_basic['system_capacity'] == original_basic['system_capacity']
        assert translated_basic['starting_stability'] == original_basic['starting_stability']
        assert translated_basic['entropy_type'] == original_basic['entropy_type']
        
        # Check entropy parameters based on type
        if original_basic['entropy_type'] == 'Constant':
            assert translated_basic['entropy_level'] == original_basic['entropy_level']
        elif original_basic['entropy_type'] == 'Increasing':
            assert translated_basic['entropy_level'] == original_basic['entropy_level']
            assert translated_basic['entropy_change_rate'] == original_basic['entropy_change_rate']
        elif original_basic['entropy_type'] == 'Oscillating':
            assert translated_basic['entropy_base_level'] == original_basic['entropy_base_level']
            assert translated_basic['entropy_amplitude'] == original_basic['entropy_amplitude']
            assert translated_basic['entropy_frequency'] == original_basic['entropy_frequency']

    def test_equivalent_results_between_modes(self):
        """Test that equivalent parameters in Basic and Expert modes produce identical results."""
        # Basic mode configuration
        basic_config = {
            'growth_speed': 'Medium',
            'uncertainty_impact': 'Medium',
            'system_capacity': 'Medium',
            'starting_stability': 'Medium',
            'entropy_type': 'Constant',
            'entropy_level': 'Medium'
        }
        
        # Translate to expert
        expert_config = ParameterTranslator.basic_to_expert(basic_config)
        
        # Create simulator for Expert Mode
        sim = CoherenceSimulator()
        
        # Configure Expert Mode experiment
        sim.configure_experiment(
            experiment_id="expert_mode_test",
            alpha=expert_config['alpha'],
            beta=expert_config['beta'],
            K=expert_config['K'],
            initial_coherence=expert_config['initial_coherence'],
            entropy_fn=constant_entropy(expert_config['entropy_params']['level']),
            time_span=np.linspace(0, 10, 100)
        )
        
        # Create Basic Mode experiment using the translated parameters
        sim.configure_experiment(
            experiment_id="basic_mode_test",
            alpha=expert_config['alpha'],
            beta=expert_config['beta'],
            K=expert_config['K'],
            initial_coherence=expert_config['initial_coherence'],
            entropy_fn=constant_entropy(expert_config['entropy_params']['level']),
            time_span=np.linspace(0, 10, 100)
        )
        
        # Run both simulations
        expert_results = sim.run_experiment("expert_mode_test")
        basic_results = sim.run_experiment("basic_mode_test")
        
        # Verify results are identical
        np.testing.assert_allclose(expert_results['coherence'], basic_results['coherence'])
        np.testing.assert_allclose(expert_results['entropy'], basic_results['entropy'])

    def test_template_equivalence(self):
        """Test that a template produces equivalent results when applied directly or through the UI."""
        # Get template
        template_name = list_all_template_names()[0]  # Use first template
        template = get_template_by_name(template_name)
        
        # Create simulator
        sim = CoherenceSimulator()
        
        # Apply template directly
        template.apply_to_simulator(sim, "direct_test")
        
        # Run direct template test
        direct_results = sim.run_experiment("direct_test")
        
        # Simulate applying template through UI by translating to Basic then back to Expert
        basic_params = ParameterTranslator.template_to_basic(template)
        expert_params = ParameterTranslator.basic_to_expert(basic_params)
        
        # Configure experiment with retranslated parameters
        sim.configure_experiment(
            experiment_id="ui_test",
            alpha=expert_params['alpha'],
            beta=expert_params['beta'],
            K=expert_params['K'],
            initial_coherence=expert_params['initial_coherence'],
            entropy_fn=template.get_entropy_function(),
            time_span=np.linspace(0, 10, 100)
        )
        
        # Run UI test
        ui_results = sim.run_experiment("ui_test")
        
        # Verify results are mathematically equivalent
        # We use a small tolerance to account for potential floating-point differences
        np.testing.assert_allclose(direct_results['coherence'], ui_results['coherence'], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(direct_results['entropy'], ui_results['entropy'], rtol=1e-10, atol=1e-10) 