"""
Tests for the core CoherenceModel implementation.

These tests validate that the coherence accumulation, entropy dynamics,
and threshold calculations correctly implement the ΨC Principle mathematics.
"""

import sys
import os
import pytest
import numpy as np
from pathlib import Path

# Add the parent directory to path so we can import our modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.models.coherence_model import CoherenceModel, shannon_entropy, memory_weight

class TestCoherenceModel:
    """Test suite for the CoherenceModel class."""
    
    def test_initialization(self):
        """Test that the model initializes with correct parameters."""
        model = CoherenceModel(alpha=0.2, K=1.5, beta=0.3, initial_coherence=0.4)
        
        assert model.alpha == 0.2
        assert model.K == 1.5
        assert model.beta == 0.3
        assert model.initial_coherence == 0.4
        assert model.results is None
    
    def test_logistic_growth_without_entropy(self):
        """Test that coherence follows logistic growth when entropy is zero."""
        # Create model with zero entropy influence
        model = CoherenceModel(alpha=0.1, K=1.0, beta=0, entropy_fn=lambda t: 0)
        time_span = np.linspace(0, 100, 500)
        
        results = model.simulate(time_span)
        
        # In pure logistic growth, should approach carrying capacity K
        assert results['coherence'][-1] > 0.99 * model.K
        
        # Should have monotonic increasing coherence
        assert np.all(np.diff(results['coherence']) >= -1e-10)  # Allow for tiny numerical errors
    
    def test_entropy_influence(self):
        """Test that entropy reduces coherence accumulation."""
        # Model without entropy
        model_no_entropy = CoherenceModel(alpha=0.1, K=1.0, beta=0, entropy_fn=lambda t: 0)
        
        # Model with constant entropy
        model_with_entropy = CoherenceModel(alpha=0.1, K=1.0, beta=0.5, entropy_fn=lambda t: 0.3)
        
        time_span = np.linspace(0, 100, 500)
        results_no_entropy = model_no_entropy.simulate(time_span)
        results_with_entropy = model_with_entropy.simulate(time_span)
        
        # Coherence with entropy should be lower than without entropy
        assert results_with_entropy['coherence'][-1] < results_no_entropy['coherence'][-1]
    
    def test_beta_scaling(self):
        """Test that the beta parameter correctly scales entropy influence."""
        # Create models with different beta values
        low_beta = CoherenceModel(alpha=0.1, K=1.0, beta=0.1, entropy_fn=lambda t: 0.3)
        high_beta = CoherenceModel(alpha=0.1, K=1.0, beta=0.5, entropy_fn=lambda t: 0.3)
        
        time_span = np.linspace(0, 100, 500)
        results_low_beta = low_beta.simulate(time_span)
        results_high_beta = high_beta.simulate(time_span)
        
        # Higher beta should result in lower coherence
        assert results_high_beta['coherence'][-1] < results_low_beta['coherence'][-1]
    
    def test_threshold_calculation(self):
        """Test that threshold calculation follows the formula."""
        # Create model with sinusoidal entropy for non-zero variance
        model = CoherenceModel(alpha=0.1, K=1.0, beta=0.2, 
                              entropy_fn=lambda t: 0.3 + 0.1 * np.sin(t))
        
        time_span = np.linspace(0, 100, 500)
        model.simulate(time_span)
        
        # Calculate threshold manually
        expected_entropy = np.mean(model.results['entropy'])
        entropy_variance = np.var(model.results['entropy'])
        lambda_theta = 1.0  # Default value in the model
        expected_threshold = expected_entropy + lambda_theta * np.sqrt(entropy_variance)
        
        # Compare with model's calculation
        threshold = model.calculate_threshold()
        
        assert np.isclose(threshold, expected_threshold, rtol=1e-10)
    
    def test_phase_transition(self):
        """Test identifying phase transitions when coherence crosses threshold."""
        # Create a model with oscillating entropy
        model = CoherenceModel(
            alpha=0.15,  # Higher alpha to ensure crossing threshold
            K=1.0,
            beta=0.1,   # Lower beta to allow coherence growth
            initial_coherence=0.5,
            entropy_fn=lambda t: 0.2 + 0.05 * np.sin(0.1 * t)  # Low oscillating entropy
        )
        
        time_span = np.linspace(0, 100, 500)
        model.simulate(time_span)
        threshold = model.calculate_threshold()
        
        # Check if coherence crosses the threshold (transitions to coherent state)
        coherence_final = model.results['coherence'][-1]
        assert coherence_final > threshold, f"Coherence {coherence_final} should exceed threshold {threshold}"

    def test_coherence_accumulation_differential_equation(self):
        """Test that the coherence accumulation follows the differential equation."""
        alpha = 0.1
        K = 1.0
        beta = 0.2
        model = CoherenceModel(alpha=alpha, K=K, beta=beta, entropy_fn=lambda t: 0.3)
        
        # Test a single step of the differential equation
        C = 0.5  # Current coherence
        t = 10   # Current time
        
        # Calculate dC/dt according to the model's equation
        dCdt_model = model.coherence_accumulation(C, t)
        
        # Calculate dC/dt manually according to the equation:
        # dC/dt = α * C * (1 - C/K) - β * H
        H = 0.3  # Our constant entropy function
        dCdt_expected = alpha * C * (1 - C/K) - beta * H
        
        assert np.isclose(dCdt_model, dCdt_expected, rtol=1e-10)
    
    def test_entropy_functions(self):
        """Test different entropy functions and their effects."""
        # Define different entropy functions
        constant_entropy = lambda t: 0.3
        increasing_entropy = lambda t: 0.1 + 0.002 * t
        decreasing_entropy = lambda t: max(0.1, 0.5 - 0.004 * t)
        oscillating_entropy = lambda t: 0.3 + 0.1 * np.sin(0.1 * t)
        
        # Create models with different entropy functions
        model_constant = CoherenceModel(entropy_fn=constant_entropy)
        model_increasing = CoherenceModel(entropy_fn=increasing_entropy)
        model_decreasing = CoherenceModel(entropy_fn=decreasing_entropy)
        model_oscillating = CoherenceModel(entropy_fn=oscillating_entropy)
        
        time_span = np.linspace(0, 100, 500)
        
        # Run simulations
        results_constant = model_constant.simulate(time_span)
        results_increasing = model_increasing.simulate(time_span)
        results_decreasing = model_decreasing.simulate(time_span)
        results_oscillating = model_oscillating.simulate(time_span)
        
        # Validate entropy values follow the expected patterns
        # Constant entropy should be constant
        entropy_std = np.std(results_constant['entropy'])
        assert entropy_std < 1e-10, f"Constant entropy should have zero standard deviation, got {entropy_std}"
        
        # Increasing entropy should be strictly increasing
        assert np.all(np.diff(results_increasing['entropy']) > 0)
        
        # Decreasing entropy should be strictly decreasing
        assert np.all(np.diff(results_decreasing['entropy']) < 0)
        
        # For oscillating entropy, check if min and max are different enough
        osc_min = np.min(results_oscillating['entropy'])
        osc_max = np.max(results_oscillating['entropy'])
        assert osc_max - osc_min > 0.1, "Oscillating entropy should have significant variation"


class TestHelperFunctions:
    """Test suite for helper functions in the coherence model module."""
    
    def test_shannon_entropy(self):
        """Test the Shannon entropy calculation."""
        # Uniform distribution should have maximum entropy
        uniform_probs = np.array([0.25, 0.25, 0.25, 0.25])
        max_entropy = shannon_entropy(uniform_probs)
        assert np.isclose(max_entropy, 2.0, rtol=1e-10)  # log2(4) = 2
        
        # Delta distribution should have zero entropy
        delta_probs = np.array([1.0, 0.0, 0.0, 0.0])
        min_entropy = shannon_entropy(delta_probs)
        assert np.isclose(min_entropy, 0.0, rtol=1e-10)
        
        # Skewed distribution should have intermediate entropy
        skewed_probs = np.array([0.5, 0.25, 0.15, 0.1])
        skewed_entropy = shannon_entropy(skewed_probs)
        assert 0 < skewed_entropy < max_entropy
    
    def test_memory_weight(self):
        """Test the memory weight calculation."""
        # Create two memory vectors
        m1 = np.array([1.0, 0.0, 0.0])
        m2 = np.array([0.0, 1.0, 0.0])
        
        # Orthogonal vectors should have zero similarity
        weight = memory_weight(m1, m2, t_i=0, t_j=0)
        assert np.isclose(weight, 0.0, atol=1e-10)
        
        # Same vector should have perfect similarity
        weight_same = memory_weight(m1, m1, t_i=0, t_j=0)
        assert np.isclose(weight_same, 1.0, rtol=1e-10)
        
        # Test time decay
        weight_time_decay = memory_weight(m1, m1, t_i=0, t_j=10, alpha=0.1)
        assert weight_time_decay < 1.0
        
        # Test with co-reflection frequency
        weight_freq = memory_weight(m1, m1, t_i=0, t_j=0, f_ij=2.0)
        assert np.isclose(weight_freq, 2.0, rtol=1e-10) 