"""
Predefined experiment templates for the Reflective Coherence Explorer.

This module provides a set of scientifically validated experiment templates
that demonstrate specific aspects of coherence dynamics. Each template is
carefully designed to illustrate particular behaviors of the system with
parameters chosen based on rigorous mathematical analysis.

All templates adhere to the mathematical foundations defined in UNDERLYING_MATH.md
and parameter ranges specified in PARAMETER_MAPPING.md.
"""

import uuid
import numpy as np
from typing import Dict, Any, Optional, Callable

class ExperimentTemplate:
    """
    A predefined experiment configuration with validated parameters.
    
    Each template encapsulates a set of scientifically sound parameters
    designed to demonstrate specific coherence behaviors.
    """
    
    def __init__(self, name: str, parameters: Dict[str, Any], description: str):
        """
        Initialize a new experiment template.
        
        Args:
            name: Name of the template
            parameters: Dictionary of experiment parameters
            description: Detailed description of what the template demonstrates
            
        Raises:
            ValueError: If parameters are outside valid scientific ranges
        """
        self.name = name
        self.parameters = parameters
        self.description = description
        self.verify()
    
    def verify(self) -> None:
        """
        Verify that all parameters are within valid scientific ranges.
        
        Raises:
            ValueError: If any parameter is outside its valid range
        """
        # Core parameter validation
        alpha = self.parameters.get("alpha")
        if not (0 < alpha < 0.5):
            raise ValueError(f"Alpha must be between 0 and 0.5, got {alpha}")
            
        beta = self.parameters.get("beta")
        if not (0 <= beta <= 1.0):
            raise ValueError(f"Beta must be between 0 and 1.0, got {beta}")
            
        K = self.parameters.get("K")
        if not (0 < K <= 2.0):
            raise ValueError(f"K must be between 0 and 2.0, got {K}")
            
        initial_coherence = self.parameters.get("initial_coherence")
        if not (0 < initial_coherence < 1.0):
            raise ValueError(
                f"Initial coherence must be between 0 and 1.0, got {initial_coherence}"
            )
    
    def apply_to_simulator(self, simulator, experiment_id=None) -> str:
        """
        Apply this template to a simulator instance.
        
        Args:
            simulator: A CoherenceSimulator instance
            experiment_id: Optional experiment ID, generated if not provided
            
        Returns:
            The experiment_id for the configured experiment
        """
        # Generate a unique ID for this experiment if not provided
        if experiment_id is None:
            experiment_id = f"template_{self.name}_{uuid.uuid4().hex[:6]}"
        
        # Configure the experiment with our parameters
        simulator.configure_experiment(
            experiment_id=experiment_id,
            **self.parameters,
            description=self.description
        )
        
        return experiment_id
        
    def get_entropy_function(self) -> Callable:
        """
        Get the entropy function from this template.
        
        Returns:
            The entropy function defined in the template parameters
        """
        return self.parameters.get("entropy_fn")


# Define predefined templates based on PARAMETER_MAPPING.md
TEMPLATES: Dict[str, ExperimentTemplate] = {}

# Template 1: Adaptation Test
TEMPLATES["Adaptation Test"] = ExperimentTemplate(
    name="Adaptation Test",
    parameters={
        "alpha": 0.15,
        "beta": 0.3,
        "K": 1.0,
        "initial_coherence": 0.6,
        "entropy_fn": lambda t: 0.1 + 0.002 * t,
        "time_span": np.linspace(0, 300, 600)
    },
    description="""
    Tests how a system maintains coherence as environmental complexity gradually increases.
    
    This experiment shows a system's ability to adapt to steadily increasing entropy,
    similar to how a student might maintain understanding while learning progressively
    more challenging material. The moderate growth rate (α=0.15) allows the system to
    adapt without being too responsive, while the entropy influence (β=0.3) ensures
    that the increasing complexity has a noticeable effect.
    """
)

# Template 2: Resilience Study
TEMPLATES["Resilience Study"] = ExperimentTemplate(
    name="Resilience Study",
    parameters={
        "alpha": 0.2,
        "beta": 0.25,
        "K": 1.0,
        "initial_coherence": 0.7,
        "entropy_fn": lambda t: 0.2 + (0.5 if (50 <= t <= 60 or 150 <= t <= 160) else 0),
        "time_span": np.linspace(0, 250, 500)
    },
    description="""
    Tests a system's recovery capabilities after entropy disturbances.
    
    This experiment demonstrates how quickly a system can recover from sudden
    spikes in entropy, similar to how an organization might respond to unexpected
    market shocks. The higher growth rate (α=0.2) allows observation of rapid
    recovery, while the moderate entropy influence (β=0.25) ensures the disturbances
    have a clear impact without overwhelming the system.
    """
)

# Template 3: Stability Analysis
TEMPLATES["Stability Analysis"] = ExperimentTemplate(
    name="Stability Analysis",
    parameters={
        "alpha": 0.1,
        "beta": 0.2,
        "K": 1.0,
        "initial_coherence": 0.5,
        "entropy_fn": lambda t: 0.3,
        "time_span": np.linspace(0, 200, 400)
    },
    description="""
    Examines steady-state behavior under constant environmental conditions.
    
    This experiment reveals how a system stabilizes under unchanging entropy,
    similar to a mature ecosystem in a stable environment. The standard growth
    rate (α=0.1) and entropy influence (β=0.2) show baseline behavior, while
    starting at the midpoint (initial_coherence=0.5) allows natural equilibrium
    to be observed.
    """
)

# Template 4: Phase Transition Explorer
TEMPLATES["Phase Transition Explorer"] = ExperimentTemplate(
    name="Phase Transition Explorer",
    parameters={
        "alpha": 0.12,
        "beta": 0.4,
        "K": 1.0,
        "initial_coherence": 0.5,
        "entropy_fn": lambda t: 0.3 + 0.2 * np.sin(0.05 * t),
        "time_span": np.linspace(0, 400, 800)
    },
    description="""
    Demonstrates transitions between coherent and incoherent states.
    
    This experiment illustrates how a system crosses critical thresholds between
    coherent and incoherent states, similar to phase transitions in physical systems
    (like water changing between liquid and gas). The oscillating entropy creates
    cycles that cross the coherence threshold, while the higher entropy influence (β=0.4)
    makes these transitions more pronounced and easier to observe.
    """
)


def get_template_by_name(template_name: str) -> Optional[ExperimentTemplate]:
    """
    Get a template by name.
    
    Args:
        template_name: The name of the template to retrieve
        
    Returns:
        The template if found, None otherwise
    """
    return TEMPLATES.get(template_name)


def get_all_templates() -> Dict[str, ExperimentTemplate]:
    """
    Get all available templates.
    
    Returns:
        Dictionary of all templates
    """
    return TEMPLATES


def list_all_template_names() -> list:
    """
    Get a list of all template names.
    
    Returns:
        List of template names
    """
    return list(TEMPLATES.keys()) 