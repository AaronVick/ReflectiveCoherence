"""
Parameter translation between Basic Mode and Expert Mode.

This module implements the bidirectional translation between simplified UI parameters
in Basic Mode and precise scientific parameters in Expert Mode, ensuring that the
underlying mathematical model remains unaltered while providing a more accessible interface.

The mapping is based on the specifications in PARAMETER_MAPPING.md, ensuring complete
traceability between simplified elements and their scientific counterparts.
"""

import random
import numpy as np
from typing import Dict, Tuple, Optional, Callable, Any, Union

class ParameterTranslator:
    """
    Translates between simplified UI parameters and scientific mathematical parameters.
    
    All translations preserve the mathematical integrity of the model by mapping
    simplified categorical values to specific numerical ranges defined in the
    parameter mapping documentation.
    """
    
    # Core parameter mappings from PARAMETER_MAPPING.md
    GROWTH_SPEED_MAP: Dict[str, Tuple[float, float]] = {
        "Low": (0.01, 0.05),
        "Medium": (0.1, 0.15),
        "High": (0.2, 0.3)
    }
    
    UNCERTAINTY_IMPACT_MAP: Dict[str, Tuple[float, float]] = {
        "Low": (0.05, 0.15),
        "Medium": (0.2, 0.3),
        "High": (0.4, 0.5)
    }
    
    SYSTEM_CAPACITY_MAP: Dict[str, Tuple[float, float]] = {
        "Small": (0.5, 0.8),
        "Medium": (1.0, 1.0),
        "Large": (1.2, 2.0)
    }
    
    STARTING_STABILITY_MAP: Dict[str, Tuple[float, float]] = {
        "Low": (0.1, 0.3),
        "Medium": (0.4, 0.6),
        "High": (0.7, 0.9)
    }
    
    # Entropy function mappings
    ENTROPY_TYPE_MAP: Dict[str, str] = {
        "Steady Environment": "Constant",
        "Increasing Complexity": "Increasing",
        "Learning Process": "Decreasing",
        "Cyclical Challenges": "Oscillating",
        "Random Environment": "Random"
    }
    
    # Inverse mapping for lookup
    INVERSE_ENTROPY_MAP: Dict[str, str] = {
        v: k for k, v in ENTROPY_TYPE_MAP.items()
    }
    
    # Parameter name mapping
    PARAMETER_NAME_MAP: Dict[str, str] = {
        "System Growth Speed": "alpha",
        "Uncertainty Impact": "beta",
        "System Capacity": "K",
        "Starting Stability": "initial_coherence",
        "Environmental Pattern": "entropy_type"
    }
    
    INVERSE_PARAMETER_NAME_MAP: Dict[str, str] = {
        v: k for k, v in PARAMETER_NAME_MAP.items()
    }
    
    @staticmethod
    def basic_to_expert(parameter_name: str, basic_value: str) -> float:
        """
        Convert a Basic Mode parameter value to its Expert Mode equivalent.
        
        Args:
            parameter_name: The name of the parameter in Basic Mode
            basic_value: The simplified value in Basic Mode
            
        Returns:
            The precise mathematical parameter value for Expert Mode
            
        Raises:
            ValueError: If the parameter name or value is not recognized
        """
        if parameter_name == "System Growth Speed":
            if basic_value in ParameterTranslator.GROWTH_SPEED_MAP:
                min_val, max_val = ParameterTranslator.GROWTH_SPEED_MAP[basic_value]
                # For Medium capacity which has exact value of 1.0
                if min_val == max_val:
                    return min_val
                return random.uniform(min_val, max_val)
                
        elif parameter_name == "Uncertainty Impact":
            if basic_value in ParameterTranslator.UNCERTAINTY_IMPACT_MAP:
                min_val, max_val = ParameterTranslator.UNCERTAINTY_IMPACT_MAP[basic_value]
                return random.uniform(min_val, max_val)
                
        elif parameter_name == "System Capacity":
            if basic_value in ParameterTranslator.SYSTEM_CAPACITY_MAP:
                min_val, max_val = ParameterTranslator.SYSTEM_CAPACITY_MAP[basic_value]
                if min_val == max_val:  # Special case for Medium
                    return min_val
                return random.uniform(min_val, max_val)
                
        elif parameter_name == "Starting Stability":
            if basic_value in ParameterTranslator.STARTING_STABILITY_MAP:
                min_val, max_val = ParameterTranslator.STARTING_STABILITY_MAP[basic_value]
                return random.uniform(min_val, max_val)
                
        elif parameter_name == "Environmental Pattern":
            if basic_value in ParameterTranslator.ENTROPY_TYPE_MAP:
                return ParameterTranslator.ENTROPY_TYPE_MAP[basic_value]
        
        raise ValueError(f"Unknown parameter '{parameter_name}' or value '{basic_value}'")
    
    @staticmethod
    def expert_to_basic(parameter_name: str, expert_value: Union[float, str]) -> Optional[str]:
        """
        Convert an Expert Mode parameter value to its Basic Mode equivalent.
        
        Args:
            parameter_name: The name of the parameter in Expert Mode
            expert_value: The precise scientific value in Expert Mode
            
        Returns:
            The simplified categorical value for Basic Mode, or None if outside defined ranges
        """
        if parameter_name == "alpha":
            for basic_name, (min_val, max_val) in ParameterTranslator.GROWTH_SPEED_MAP.items():
                if min_val <= expert_value <= max_val:
                    return basic_name
                    
        elif parameter_name == "beta":
            for basic_name, (min_val, max_val) in ParameterTranslator.UNCERTAINTY_IMPACT_MAP.items():
                if min_val <= expert_value <= max_val:
                    return basic_name
                    
        elif parameter_name == "K":
            for basic_name, (min_val, max_val) in ParameterTranslator.SYSTEM_CAPACITY_MAP.items():
                if min_val <= expert_value <= max_val:
                    return basic_name
                    
        elif parameter_name == "initial_coherence":
            for basic_name, (min_val, max_val) in ParameterTranslator.STARTING_STABILITY_MAP.items():
                if min_val <= expert_value <= max_val:
                    return basic_name
                    
        elif parameter_name == "entropy_type" and isinstance(expert_value, str):
            if expert_value in ParameterTranslator.INVERSE_ENTROPY_MAP:
                return ParameterTranslator.INVERSE_ENTROPY_MAP[expert_value]
        
        return None
    
    @staticmethod
    def translate_entropy_params(
        basic_pattern: str, 
        basic_params: Dict[str, Any]
    ) -> Tuple[str, Callable]:
        """
        Translate Basic Mode entropy pattern and parameters to Expert Mode entropy function.
        
        Args:
            basic_pattern: The simplified entropy pattern name
            basic_params: The simplified parameters for the pattern
            
        Returns:
            A tuple of (entropy_type, entropy_function) for Expert Mode
            
        Raises:
            ValueError: If the pattern is not recognized
        """
        if basic_pattern == "Steady Environment":
            level = basic_params.get("Steady Level", "Medium")
            value_map = {"Low": 0.1, "Medium": 0.3, "High": 0.5}
            value = value_map.get(level, 0.3)
            return "Constant", lambda t: value
            
        elif basic_pattern == "Increasing Complexity":
            growth = basic_params.get("Complexity Growth", "Medium")
            start = 0.1
            rate_map = {"Slow": 0.001, "Medium": 0.003, "Fast": 0.005}
            rate = rate_map.get(growth, 0.003)
            return "Increasing", lambda t: start + rate * t
            
        elif basic_pattern == "Learning Process":
            speed = basic_params.get("Learning Rate", "Medium")
            start = 0.5
            rate_map = {"Slow": 0.001, "Medium": 0.003, "Fast": 0.005}
            rate = rate_map.get(speed, 0.003)
            return "Decreasing", lambda t: max(0.1, start - rate * t)
            
        elif basic_pattern == "Cyclical Challenges":
            freq = basic_params.get("Challenge Frequency", "Medium")
            intensity = basic_params.get("Challenge Intensity", "Moderate")
            
            base = 0.3
            freq_map = {"Low": 0.05, "Medium": 0.1, "High": 0.2}
            amp_map = {"Mild": 0.1, "Moderate": 0.2, "Severe": 0.3}
            
            frequency = freq_map.get(freq, 0.1)
            amplitude = amp_map.get(intensity, 0.2)
            
            return "Oscillating", lambda t: base + amplitude * np.sin(frequency * t)
            
        elif basic_pattern == "Random Environment":
            level = basic_params.get("Uncertainty Level", "Medium")
            variability = basic_params.get("Variability", "Medium")
            
            mean_map = {"Low": 0.2, "Medium": 0.3, "High": 0.4}
            std_map = {"Low": 0.05, "Medium": 0.1, "High": 0.2}
            
            mean = mean_map.get(level, 0.3)
            std = std_map.get(variability, 0.1)
            
            # Note: We need to set the random seed to ensure reproducibility
            random_state = np.random.RandomState(42)
            return "Random", lambda t: np.clip(
                mean + std * random_state.randn(len(t) if hasattr(t, '__len__') else 1),
                0.05, 0.95  # Clamp between reasonable bounds
            )
        
        raise ValueError(f"Unknown entropy pattern: {basic_pattern}")
    
    @staticmethod
    def get_parameter_midpoint(parameter_name: str, basic_value: str) -> float:
        """
        Get the midpoint value for a parameter range.
        
        This is useful when you want a reproducible value rather than a random one.
        
        Args:
            parameter_name: The name of the parameter in Basic Mode
            basic_value: The simplified value in Basic Mode
            
        Returns:
            The midpoint of the parameter range
            
        Raises:
            ValueError: If the parameter name or value is not recognized
        """
        if parameter_name == "System Growth Speed":
            if basic_value in ParameterTranslator.GROWTH_SPEED_MAP:
                min_val, max_val = ParameterTranslator.GROWTH_SPEED_MAP[basic_value]
                return (min_val + max_val) / 2
                
        elif parameter_name == "Uncertainty Impact":
            if basic_value in ParameterTranslator.UNCERTAINTY_IMPACT_MAP:
                min_val, max_val = ParameterTranslator.UNCERTAINTY_IMPACT_MAP[basic_value]
                return (min_val + max_val) / 2
                
        elif parameter_name == "System Capacity":
            if basic_value in ParameterTranslator.SYSTEM_CAPACITY_MAP:
                min_val, max_val = ParameterTranslator.SYSTEM_CAPACITY_MAP[basic_value]
                return (min_val + max_val) / 2
                
        elif parameter_name == "Starting Stability":
            if basic_value in ParameterTranslator.STARTING_STABILITY_MAP:
                min_val, max_val = ParameterTranslator.STARTING_STABILITY_MAP[basic_value]
                return (min_val + max_val) / 2
        
        raise ValueError(f"Unknown parameter '{parameter_name}' or value '{basic_value}'")
    
    @staticmethod
    def translate_full_config(basic_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate a complete Basic Mode configuration to Expert Mode.
        
        Args:
            basic_config: Dictionary containing all Basic Mode parameters
            
        Returns:
            Dictionary with equivalent Expert Mode parameters
            
        Raises:
            ValueError: If any parameter cannot be translated
        """
        expert_config = {}
        
        # Translate core parameters
        for basic_name, expert_name in ParameterTranslator.PARAMETER_NAME_MAP.items():
            if basic_name in basic_config:
                # Use midpoint for reproducibility
                if expert_name != "entropy_type":
                    expert_config[expert_name] = ParameterTranslator.get_parameter_midpoint(
                        basic_name, basic_config[basic_name]
                    )
        
        # Handle entropy separately
        if "Environmental Pattern" in basic_config:
            pattern = basic_config["Environmental Pattern"]
            params = basic_config.get("Environmental Parameters", {})
            
            entropy_type, entropy_fn = ParameterTranslator.translate_entropy_params(pattern, params)
            expert_config["entropy_type"] = entropy_type
            expert_config["entropy_fn"] = entropy_fn
        
        # Copy other parameters directly
        for key in ["experiment_id", "description", "time_span"]:
            if key in basic_config:
                expert_config[key] = basic_config[key]
        
        return expert_config 