"""
Tests for Parameter Translation Layer.

These tests verify that the translation between simplified UI parameters and 
scientific mathematical parameters is accurate and maintains complete integrity.
"""

import sys
import os
import pytest
import numpy as np
import random
from pathlib import Path
import unittest

# Add the parent directory to path so we can import our modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Once implemented, these will be the actual imports
# from app.translation.parameter_mapping import ParameterTranslator
# from app.templates.experiment_templates import ExperimentTemplate, TEMPLATES

# For now, we'll define expected values based on our documentation
EXPECTED_MAPPING = {
    "GROWTH_SPEED_MAP": {
        "Low": (0.01, 0.05),
        "Medium": (0.1, 0.15),
        "High": (0.2, 0.3)
    },
    "UNCERTAINTY_IMPACT_MAP": {
        "Low": (0.05, 0.15),
        "Medium": (0.2, 0.3),
        "High": (0.4, 0.5)
    },
    "SYSTEM_CAPACITY_MAP": {
        "Small": (0.5, 0.8),
        "Medium": (1.0, 1.0),
        "Large": (1.2, 2.0)
    },
    "STARTING_STABILITY_MAP": {
        "Low": (0.1, 0.3),
        "Medium": (0.4, 0.6),
        "High": (0.7, 0.9)
    }
}

EXPECTED_ENTROPY_MAPPING = {
    "Steady Environment": "Constant",
    "Increasing Complexity": "Increasing",
    "Learning Process": "Decreasing",
    "Cyclical Challenges": "Oscillating",
    "Random Environment": "Random"
}

# Mock translator until actual implementation
class MockParameterTranslator:
    GROWTH_SPEED_MAP = EXPECTED_MAPPING["GROWTH_SPEED_MAP"]
    UNCERTAINTY_IMPACT_MAP = EXPECTED_MAPPING["UNCERTAINTY_IMPACT_MAP"]
    SYSTEM_CAPACITY_MAP = EXPECTED_MAPPING["SYSTEM_CAPACITY_MAP"]
    STARTING_STABILITY_MAP = EXPECTED_MAPPING["STARTING_STABILITY_MAP"]
    
    @staticmethod
    def basic_to_expert(parameter_name, basic_value):
        """Mock implementation of basic to expert translation"""
        if parameter_name == "System Growth Speed":
            if basic_value in MockParameterTranslator.GROWTH_SPEED_MAP:
                min_val, max_val = MockParameterTranslator.GROWTH_SPEED_MAP[basic_value]
                return random.uniform(min_val, max_val)
        elif parameter_name == "Uncertainty Impact":
            if basic_value in MockParameterTranslator.UNCERTAINTY_IMPACT_MAP:
                min_val, max_val = MockParameterTranslator.UNCERTAINTY_IMPACT_MAP[basic_value]
                return random.uniform(min_val, max_val)
        elif parameter_name == "System Capacity":
            if basic_value in MockParameterTranslator.SYSTEM_CAPACITY_MAP:
                min_val, max_val = MockParameterTranslator.SYSTEM_CAPACITY_MAP[basic_value]
                if min_val == max_val:  # Special case for Medium
                    return min_val
                return random.uniform(min_val, max_val)
        elif parameter_name == "Starting Stability":
            if basic_value in MockParameterTranslator.STARTING_STABILITY_MAP:
                min_val, max_val = MockParameterTranslator.STARTING_STABILITY_MAP[basic_value]
                return random.uniform(min_val, max_val)
        
        raise ValueError(f"Unknown parameter '{parameter_name}' or value '{basic_value}'")
    
    @staticmethod
    def expert_to_basic(parameter_name, expert_value):
        """Mock implementation of expert to basic translation"""
        if parameter_name == "alpha":
            for basic_name, (min_val, max_val) in MockParameterTranslator.GROWTH_SPEED_MAP.items():
                if min_val <= expert_value <= max_val:
                    return basic_name
        elif parameter_name == "beta":
            for basic_name, (min_val, max_val) in MockParameterTranslator.UNCERTAINTY_IMPACT_MAP.items():
                if min_val <= expert_value <= max_val:
                    return basic_name
        elif parameter_name == "K":
            for basic_name, (min_val, max_val) in MockParameterTranslator.SYSTEM_CAPACITY_MAP.items():
                if min_val <= expert_value <= max_val:
                    return basic_name
        elif parameter_name == "initial_coherence":
            for basic_name, (min_val, max_val) in MockParameterTranslator.STARTING_STABILITY_MAP.items():
                if min_val <= expert_value <= max_val:
                    return basic_name
        
        return None  # For values outside the predefined ranges


class TestParameterTranslation:
    """Test suite for parameter translation between Basic and Expert modes."""
    
    @pytest.fixture
    def translator(self):
        """Fixture to provide a translator instance for tests."""
        # In the future, return the actual ParameterTranslator
        return MockParameterTranslator
    
    def test_basic_to_expert_translation(self, translator):
        """Test translation from basic UI terms to expert mathematical parameters."""
        # System Growth Speed (alpha)
        alpha_low = translator.basic_to_expert("System Growth Speed", "Low")
        assert 0.01 <= alpha_low <= 0.05, "Low growth speed should translate to alpha between 0.01-0.05"
        
        alpha_medium = translator.basic_to_expert("System Growth Speed", "Medium")
        assert 0.1 <= alpha_medium <= 0.15, "Medium growth speed should translate to alpha between 0.1-0.15"
        
        alpha_high = translator.basic_to_expert("System Growth Speed", "High")
        assert 0.2 <= alpha_high <= 0.3, "High growth speed should translate to alpha between 0.2-0.3"
        
        # Uncertainty Impact (beta)
        beta_low = translator.basic_to_expert("Uncertainty Impact", "Low")
        assert 0.05 <= beta_low <= 0.15, "Low uncertainty impact should translate to beta between 0.05-0.15"
        
        beta_medium = translator.basic_to_expert("Uncertainty Impact", "Medium")
        assert 0.2 <= beta_medium <= 0.3, "Medium uncertainty impact should translate to beta between 0.2-0.3"
        
        beta_high = translator.basic_to_expert("Uncertainty Impact", "High")
        assert 0.4 <= beta_high <= 0.5, "High uncertainty impact should translate to beta between 0.4-0.5"
        
        # System Capacity (K)
        k_small = translator.basic_to_expert("System Capacity", "Small")
        assert 0.5 <= k_small <= 0.8, "Small system capacity should translate to K between 0.5-0.8"
        
        k_medium = translator.basic_to_expert("System Capacity", "Medium")
        assert k_medium == 1.0, "Medium system capacity should translate to K of 1.0"
        
        k_large = translator.basic_to_expert("System Capacity", "Large")
        assert 1.2 <= k_large <= 2.0, "Large system capacity should translate to K between 1.2-2.0"
        
        # Starting Stability (initial_coherence)
        ic_low = translator.basic_to_expert("Starting Stability", "Low")
        assert 0.1 <= ic_low <= 0.3, "Low starting stability should translate to initial_coherence between 0.1-0.3"
        
        ic_medium = translator.basic_to_expert("Starting Stability", "Medium")
        assert 0.4 <= ic_medium <= 0.6, "Medium starting stability should translate to initial_coherence between 0.4-0.6"
        
        ic_high = translator.basic_to_expert("Starting Stability", "High")
        assert 0.7 <= ic_high <= 0.9, "High starting stability should translate to initial_coherence between 0.7-0.9"
    
    def test_expert_to_basic_translation(self, translator):
        """Test translation from expert mathematical parameters to basic UI terms."""
        # alpha to System Growth Speed
        assert translator.expert_to_basic("alpha", 0.03) == "Low"
        assert translator.expert_to_basic("alpha", 0.12) == "Medium"
        assert translator.expert_to_basic("alpha", 0.25) == "High"
        
        # beta to Uncertainty Impact
        assert translator.expert_to_basic("beta", 0.1) == "Low"
        assert translator.expert_to_basic("beta", 0.25) == "Medium"
        assert translator.expert_to_basic("beta", 0.45) == "High"
        
        # K to System Capacity
        assert translator.expert_to_basic("K", 0.7) == "Small"
        assert translator.expert_to_basic("K", 1.0) == "Medium"
        assert translator.expert_to_basic("K", 1.5) == "Large"
        
        # initial_coherence to Starting Stability
        assert translator.expert_to_basic("initial_coherence", 0.2) == "Low"
        assert translator.expert_to_basic("initial_coherence", 0.5) == "Medium"
        assert translator.expert_to_basic("initial_coherence", 0.8) == "High"
    
    def test_roundtrip_translation(self, translator):
        """Test that parameters maintain their meaning when translated back and forth."""
        # Test for each parameter
        test_parameters = [
            ("System Growth Speed", "Low", "alpha"),
            ("System Growth Speed", "Medium", "alpha"),
            ("System Growth Speed", "High", "alpha"),
            ("Uncertainty Impact", "Low", "beta"),
            ("Uncertainty Impact", "Medium", "beta"),
            ("Uncertainty Impact", "High", "beta"),
            ("System Capacity", "Small", "K"),
            ("System Capacity", "Medium", "K"),
            ("System Capacity", "Large", "K"),
            ("Starting Stability", "Low", "initial_coherence"),
            ("Starting Stability", "Medium", "initial_coherence"),
            ("Starting Stability", "High", "initial_coherence")
        ]
        
        for basic_name, basic_value, expert_name in test_parameters:
            # Translate from basic to expert
            expert_value = translator.basic_to_expert(basic_name, basic_value)
            
            # Translate back from expert to basic
            roundtrip_value = translator.expert_to_basic(expert_name, expert_value)
            
            # Should get the same basic value back
            assert roundtrip_value == basic_value, f"Roundtrip translation failed for {basic_name}={basic_value}"
    
    def test_edge_values(self, translator):
        """Test behavior at the boundaries between different parameter categories."""
        # Test edge between Low and Medium alpha
        low_max = translator.basic_to_expert("System Growth Speed", "Low")
        medium_min = translator.basic_to_expert("System Growth Speed", "Medium")
        
        assert translator.expert_to_basic("alpha", 0.049) == "Low"
        assert translator.expert_to_basic("alpha", 0.051) == None  # Gap between ranges
        assert translator.expert_to_basic("alpha", 0.101) == "Medium"
        
        # Similar tests for other parameters
        assert translator.expert_to_basic("beta", 0.149) == "Low"
        assert translator.expert_to_basic("beta", 0.151) == None  # Gap between ranges
        assert translator.expert_to_basic("beta", 0.201) == "Medium"
    
    def test_invalid_values(self, translator):
        """Test handling of invalid parameter values."""
        # Invalid basic parameter name
        with pytest.raises(ValueError):
            translator.basic_to_expert("Invalid Parameter", "Medium")
        
        # Invalid basic value
        with pytest.raises(ValueError):
            translator.basic_to_expert("System Growth Speed", "Invalid Value")
        
        # Expert value outside any defined range
        assert translator.expert_to_basic("alpha", -0.1) is None
        assert translator.expert_to_basic("alpha", 0.6) is None
    
    def test_translation_consistency(self, translator):
        """Test that the translation is consistent with our documentation."""
        # Check if the translator's mappings match the expected mappings
        assert translator.GROWTH_SPEED_MAP == EXPECTED_MAPPING["GROWTH_SPEED_MAP"]
        assert translator.UNCERTAINTY_IMPACT_MAP == EXPECTED_MAPPING["UNCERTAINTY_IMPACT_MAP"]
        assert translator.SYSTEM_CAPACITY_MAP == EXPECTED_MAPPING["SYSTEM_CAPACITY_MAP"]
        assert translator.STARTING_STABILITY_MAP == EXPECTED_MAPPING["STARTING_STABILITY_MAP"]


class TestTemplateTranslation:
    """Test suite for experiment template parameter validation."""
    
    def test_template_parameter_validity(self):
        """
        Test that all predefined templates have parameters within valid mathematical ranges.
        
        This will be expanded once the actual template implementation is available.
        """
        # Template parameters from PARAMETER_MAPPING.md
        template_params = {
            "Adaptation Test": {
                "alpha": 0.15,
                "beta": 0.3,
                "K": 1.0,
                "initial_coherence": 0.6,
                "entropy_type": "Increasing",
                "entropy_params": {"H0": 0.1, "r": 0.002}
            },
            "Resilience Study": {
                "alpha": 0.2,
                "beta": 0.25,
                "K": 1.0,
                "initial_coherence": 0.7,
                "entropy_type": "Custom",
                "entropy_params": {"base": 0.2, "spikes": [50, 150]}
            },
            "Stability Analysis": {
                "alpha": 0.1,
                "beta": 0.2,
                "K": 1.0,
                "initial_coherence": 0.5,
                "entropy_type": "Constant",
                "entropy_params": {"value": 0.3}
            },
            "Phase Transition Explorer": {
                "alpha": 0.12,
                "beta": 0.4,
                "K": 1.0,
                "initial_coherence": 0.5,
                "entropy_type": "Oscillating",
                "entropy_params": {"base": 0.3, "amplitude": 0.2, "frequency": 0.05}
            }
        }
        
        # Verify all templates have valid parameters
        for template_name, params in template_params.items():
            assert 0 < params["alpha"] < 0.5, f"Template {template_name} has invalid alpha value"
            assert 0 <= params["beta"] <= 1.0, f"Template {template_name} has invalid beta value"
            assert 0 < params["K"] <= 2.0, f"Template {template_name} has invalid K value"
            assert 0 < params["initial_coherence"] < 1.0, f"Template {template_name} has invalid initial_coherence value"
    
    def test_template_scientific_coherence(self):
        """
        Test that templates are scientifically sound and produce expected behavior.
        
        This will be expanded once the actual simulation engine can be connected.
        """
        # For now, just check a few known scientific relationships
        template_params = {
            "Adaptation Test": {"alpha": 0.15, "beta": 0.3},
            "Resilience Study": {"alpha": 0.2, "beta": 0.25},
            "Stability Analysis": {"alpha": 0.1, "beta": 0.2},
            "Phase Transition Explorer": {"alpha": 0.12, "beta": 0.4}
        }
        
        # Higher beta should generally lead to lower coherence growth
        # Higher alpha should generally lead to higher coherence growth
        # Testing this will require running the actual simulation 