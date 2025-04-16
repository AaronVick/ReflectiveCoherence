import numpy as np
import scipy.integrate as integrate
from typing import Callable, Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt

class CoherenceModel:
    """
    Implements the core mathematical model for Reflective Coherence (ΨC) dynamics.
    
    This model simulates coherence accumulation governed by logistic growth modified
    by entropy, as described in the ΨC Principle.
    """
    
    def __init__(
        self,
        alpha: float = 0.1,        # Coherence growth rate
        K: float = 1.0,            # Maximum coherence (carrying capacity)
        beta: float = 0.2,         # Entropy influence factor
        initial_coherence: float = 0.5,  # Starting coherence value
        entropy_fn: Optional[Callable] = None  # Custom entropy function
    ):
        """
        Initialize the coherence model with parameters.
        
        Args:
            alpha: Coherence growth rate parameter
            K: Maximum coherence capacity parameter
            beta: Entropy influence factor parameter
            initial_coherence: Starting coherence value
            entropy_fn: Optional custom entropy function, defaults to random normal
        """
        # Validate parameters
        self.alpha = max(0.001, alpha)  # Ensure positive growth rate
        self.K = max(0.1, K)  # Ensure positive capacity
        self.beta = max(0, beta)  # Non-negative entropy influence
        self.initial_coherence = max(1e-6, initial_coherence)  # Ensure positive initial state
        
        # Default entropy function (random normal distribution) with non-negative guard
        if entropy_fn is None:
            self.entropy_fn = lambda t: max(0, np.abs(np.random.normal(0.3, 0.1)))
        else:
            # Wrap the provided entropy function to ensure non-negative values
            self._raw_entropy_fn = entropy_fn
            self.entropy_fn = lambda t: max(0, self._raw_entropy_fn(t))
            
        self.results = None  # Will store simulation results
    
    def coherence_accumulation(self, C: float, t: float) -> float:
        """
        Core differential equation for coherence accumulation.
        
        dC(t)/dt = α * C(t) * (1 - C(t)/K) - β * H(M(t))
        
        Args:
            C: Current coherence value
            t: Current time point
            
        Returns:
            Rate of change of coherence at time t
        """
        # Ensure coherence is non-negative (avoids numerical issues)
        C = max(1e-10, C)
        
        # Get entropy with non-negative guard
        H = self.entropy_fn(t)  # Current entropy value
        
        # Calculate rate of change with boundary check
        dCdt = self.alpha * C * (1 - C / self.K) - self.beta * H
        
        # Add constraint to prevent extreme negative rates when coherence is very low
        if C < 1e-6 and dCdt < 0:
            dCdt = max(dCdt, -C)  # Prevent coherence from going negative
            
        return dCdt
    
    def simulate(self, time_span: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run the coherence simulation over the given time span.
        
        Args:
            time_span: Array of time points to simulate
            
        Returns:
            Dictionary with time, coherence, and entropy arrays
        """
        # Generate solution using scipy's ODE solver with increased precision for challenging cases
        sol = integrate.solve_ivp(
            lambda t, C: self.coherence_accumulation(C, t),
            [time_span[0], time_span[-1]],
            [self.initial_coherence],
            t_eval=time_span,
            method='RK45',  # Explicit Runge-Kutta method
            rtol=1e-6,      # Relative tolerance for adaptive step control
            atol=1e-9       # Absolute tolerance for adaptive step control
        )
        
        # Calculate entropy at each time point (with non-negative guard)
        entropy_values = np.array([self.entropy_fn(t) for t in time_span])
        
        # Apply coherence non-negative constraint (ensure physical constraint is met)
        coherence_values = np.maximum(sol.y[0], 1e-10)
        
        # Store results
        self.results = {
            'time': sol.t,
            'coherence': coherence_values,
            'entropy': entropy_values
        }
        
        return self.results
    
    def calculate_threshold(self) -> float:
        """
        Calculate the phase transition threshold.
        
        θ = E[H(M(t))] + λθ * sqrt(Var(H(M(t))))
        
        Returns:
            The calculated threshold value
        """
        if self.results is None:
            raise ValueError("Run simulation first to calculate threshold")
        
        # Using lambda_theta = 1.0 as default
        lambda_theta = 1.0
        
        # Calculate expected entropy and variance
        expected_entropy = np.mean(self.results['entropy'])
        entropy_variance = np.var(self.results['entropy'])
        
        # Calculate threshold
        threshold = expected_entropy + lambda_theta * np.sqrt(entropy_variance)
        
        return threshold
    
    def plot_results(self, show_threshold: bool = True, title: str = "Coherence Dynamics") -> plt.Figure:
        """
        Plot the simulation results.
        
        Args:
            show_threshold: Whether to show the coherence threshold line
            title: Title for the plot
            
        Returns:
            Matplotlib figure object
        """
        if self.results is None:
            raise ValueError("Run simulation first to plot results")
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot coherence
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Coherence', color='blue')
        ax1.plot(self.results['time'], self.results['coherence'], 'b-', label='Coherence')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Create second y-axis for entropy
        ax2 = ax1.twinx()
        ax2.set_ylabel('Entropy', color='red')
        ax2.plot(self.results['time'], self.results['entropy'], 'r-', label='Entropy')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add threshold if requested
        if show_threshold:
            threshold = self.calculate_threshold()
            ax1.axhline(y=threshold, color='green', linestyle='--', label=f'Threshold (θ={threshold:.2f})')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.title(title)
        plt.tight_layout()
        
        return fig

def safe_shannon_entropy(probabilities: np.ndarray) -> float:
    """
    Safely calculate Shannon entropy for a probability distribution,
    with guards against invalid inputs.
    
    H(M(t)) = -∑(i=1 to N) p(m_i) * log(p(m_i))
    
    Args:
        probabilities: Array of probability values
        
    Returns:
        Shannon entropy value
    """
    # Guard against empty arrays
    if len(probabilities) == 0:
        return 0.0
        
    # Normalize probabilities to ensure they sum to 1
    if abs(np.sum(probabilities) - 1.0) > 1e-10:
        probabilities = probabilities / np.sum(probabilities)
    
    # Filter out zero and negative probabilities to avoid log(0) or log(negative)
    valid_probs = probabilities[probabilities > 1e-10]
    
    # If no valid probabilities, return 0
    if len(valid_probs) == 0:
        return 0.0
        
    return -np.sum(valid_probs * np.log2(valid_probs))

def memory_weight(m_i: np.ndarray, m_j: np.ndarray, 
                 t_i: float, t_j: float, 
                 alpha: float = 0.1, f_ij: float = 1.0) -> float:
    """
    Calculate the weight between two memory states.
    
    w_ij = cos(z(m_i), z(m_j)) * f_ij / (1 + α|t_i - t_j|)
    
    Args:
        m_i, m_j: Memory state vectors
        t_i, t_j: Time points of memories
        alpha: Decay factor
        f_ij: Frequency of co-reflection
        
    Returns:
        Weight between memories
    """
    # Guard against zero-norm vectors
    norm_i = np.linalg.norm(m_i)
    norm_j = np.linalg.norm(m_j)
    
    if norm_i < 1e-10 or norm_j < 1e-10:
        return 0.0  # Cannot calculate meaningful similarity for zero vectors
    
    # Calculate cosine similarity
    cos_sim = np.dot(m_i, m_j) / (norm_i * norm_j)
    
    # Clamp to [-1, 1] in case of numerical errors
    cos_sim = max(-1.0, min(1.0, cos_sim))
    
    # Calculate time decay factor
    time_decay = 1.0 / (1.0 + alpha * abs(t_i - t_j))
    
    return cos_sim * f_ij * time_decay 