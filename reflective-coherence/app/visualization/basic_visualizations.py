"""
Enhanced visualizations for Basic Mode interface.

This module provides visualization tools with interpretive overlays, annotations,
and context-aware displays that help users understand coherence dynamics by highlighting
key events and transitions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Tuple, Dict, Any, Optional, Union
import streamlit as st
from matplotlib.figure import Figure
import warnings

@st.cache_data
def downsample_time_series(
    time: np.ndarray, 
    values: np.ndarray, 
    target_points: int = 500,
    preserve_features: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample time series data for more efficient plotting while preserving critical features.
    
    Args:
        time: Array of time points
        values: Array of corresponding values
        target_points: Target number of points after downsampling
        preserve_features: Whether to preserve extrema and high-variation regions
        
    Returns:
        Tuple of (downsampled_time, downsampled_values)
    """
    # Skip if already smaller than target
    if len(time) <= target_points:
        return time, values
    
    # Ensure inputs are numpy arrays
    time = np.asarray(time)
    values = np.asarray(values)
    
    # For extremely large datasets, use a two-stage approach
    if len(time) > 100000:
        # First rough downsample to a manageable size
        factor = len(time) // 50000
        time_temp = time[::factor]
        values_temp = values[::factor]
        
        # Then apply more sophisticated downsampling
        return downsample_time_series(time_temp, values_temp, target_points, preserve_features)
    
    # Calculate downsample factor
    factor = max(1, len(time) // target_points)
    
    if preserve_features:
        # Find critical points (local extrema, high curvature points)
        critical_indices = []
        
        # Always include first and last points
        critical_indices.append(0)
        critical_indices.append(len(values) - 1)
        
        # Find local minima and maxima
        for i in range(1, len(values) - 1):
            if (values[i] > values[i-1] and values[i] > values[i+1]) or \
               (values[i] < values[i-1] and values[i] < values[i+1]):
                critical_indices.append(i)
        
        # Safety check: if we have too many critical points, we need to select a subset
        if len(critical_indices) >= target_points:
            # Too many critical points - just use uniform sampling instead
            # but keep first and last points
            uniform_indices = np.linspace(0, len(time) - 1, target_points).astype(int)
            all_indices = sorted(set(uniform_indices))
        else:
            # We have room for both critical points and uniform sampling
            uniform_indices = np.linspace(0, len(time) - 1, 
                                         max(10, target_points - len(critical_indices))).astype(int)
            
            # Combine and sort indices (avoiding duplicates)
            all_indices = sorted(set(list(critical_indices) + list(uniform_indices)))
        
        # Select points
        downsampled_time = time[all_indices]
        downsampled_values = values[all_indices]
    else:
        # Simple stride-based downsampling
        downsampled_time = time[::factor]
        downsampled_values = values[::factor]
        
        # Ensure we include the last point for proper visualization
        if time[-1] != downsampled_time[-1]:
            downsampled_time = np.append(downsampled_time, time[-1])
            downsampled_values = np.append(downsampled_values, values[-1])
    
    return downsampled_time, downsampled_values

@st.cache_data
def detect_phase_transitions(
    time: np.ndarray,
    coherence: np.ndarray,
    threshold: Union[np.ndarray, float]
) -> List[float]:
    """
    Detect phase transitions where coherence crosses the threshold.
    
    Args:
        time: Array of time points
        coherence: Array of coherence values
        threshold: Array of threshold values or single threshold value
        
    Returns:
        List of time points where phase transitions occur
    """
    # Ensure inputs are numpy arrays
    time = np.asarray(time)
    coherence = np.asarray(coherence)
    
    # Guard against empty inputs
    if len(time) == 0 or len(coherence) == 0:
        return []
        
    # Ensure coherence and time arrays have the same length
    if len(time) != len(coherence):
        warnings.warn(f"Time and coherence arrays have different lengths: {len(time)} vs {len(coherence)}")
        min_len = min(len(time), len(coherence))
        time = time[:min_len]
        coherence = coherence[:min_len]
    
    # Downsample for large datasets
    if len(time) > 1000:
        time, coherence = downsample_time_series(time, coherence, 1000, preserve_features=True)
        if isinstance(threshold, np.ndarray):
            if len(threshold) > len(time):
                _, threshold = downsample_time_series(time, threshold[:len(time)], 1000)
            else:
                # Handle case where threshold array is shorter than time
                threshold_temp = np.ones_like(coherence) * np.mean(threshold)
                for i in range(min(len(threshold), len(threshold_temp))):
                    threshold_temp[i] = threshold[i]
                threshold = threshold_temp
    
    transitions = []
    
    # Convert single threshold to array if necessary
    if isinstance(threshold, (int, float)):
        threshold_arr = np.ones_like(coherence) * threshold
    else:
        # Ensure threshold array has the right length
        if len(threshold) != len(coherence):
            warnings.warn(f"Threshold and coherence arrays have different lengths: {len(threshold)} vs {len(coherence)}")
            if len(threshold) > len(coherence):
                threshold_arr = threshold[:len(coherence)]
            else:
                # Extend threshold array if it's too short
                threshold_arr = np.ones_like(coherence) * np.mean(threshold)
                threshold_arr[:len(threshold)] = threshold
        else:
            threshold_arr = threshold
    
    # Detect crossings
    try:
        for i in range(1, len(coherence)):
            # Check if coherence crossed above threshold
            if coherence[i-1] < threshold_arr[i-1] and coherence[i] >= threshold_arr[i]:
                # Linear interpolation to find exact crossing point
                if (coherence[i] - coherence[i-1]) != (threshold_arr[i] - threshold_arr[i-1]):  # Avoid division by zero
                    t_cross = time[i-1] + (time[i] - time[i-1]) * (
                        (threshold_arr[i-1] - coherence[i-1]) / 
                        ((coherence[i] - coherence[i-1]) - (threshold_arr[i] - threshold_arr[i-1]))
                    )
                    # Bound to actual time interval to avoid extrapolation errors
                    t_cross = max(time[i-1], min(time[i], t_cross))
                    transitions.append(t_cross)
                else:
                    # If slopes are identical, use midpoint
                    transitions.append((time[i] + time[i-1]) / 2)
            
            # Check if coherence crossed below threshold
            elif coherence[i-1] >= threshold_arr[i-1] and coherence[i] < threshold_arr[i]:
                # Linear interpolation to find exact crossing point
                if (coherence[i] - coherence[i-1]) != (threshold_arr[i] - threshold_arr[i-1]):  # Avoid division by zero
                    t_cross = time[i-1] + (time[i] - time[i-1]) * (
                        (threshold_arr[i-1] - coherence[i-1]) / 
                        ((coherence[i] - coherence[i-1]) - (threshold_arr[i] - threshold_arr[i-1]))
                    )
                    # Bound to actual time interval to avoid extrapolation errors
                    t_cross = max(time[i-1], min(time[i], t_cross))
                    transitions.append(t_cross)
                else:
                    # If slopes are identical, use midpoint
                    transitions.append((time[i] + time[i-1]) / 2)
    except Exception as e:
        warnings.warn(f"Error in phase transition detection: {str(e)}")
        # Return empty list on error
        return []
    
    return transitions

@st.cache_data
def identify_coherent_regions(
    time: np.ndarray,
    coherence: np.ndarray,
    threshold: Union[np.ndarray, float]
) -> List[Tuple[float, float]]:
    """
    Identify regions where the system maintains coherence above threshold.
    
    Args:
        time: Array of time points
        coherence: Array of coherence values
        threshold: Array of threshold values or single threshold value
        
    Returns:
        List of (start_time, end_time) tuples for coherent regions
    """
    # Downsample for large datasets
    if len(time) > 1000:
        time, coherence = downsample_time_series(time, coherence, 1000)
        if isinstance(threshold, np.ndarray):
            _, threshold = downsample_time_series(time, threshold, 1000)
    
    coherent_regions = []
    
    # Convert single threshold to array if necessary
    if isinstance(threshold, (int, float)):
        threshold_arr = np.ones_like(coherence) * threshold
    else:
        threshold_arr = threshold
    
    # Find regions where coherence > threshold
    in_coherent_region = False
    region_start = None
    
    for i in range(len(coherence)):
        # If coherence exceeds threshold and we're not already in a region, start new region
        if coherence[i] >= threshold_arr[i] and not in_coherent_region:
            in_coherent_region = True
            region_start = time[i]
            
            # If this is the first point, interpolate to find exact start
            if i > 0 and coherence[i-1] < threshold_arr[i-1]:
                # Linear interpolation to find exact crossing point
                t_cross = time[i-1] + (time[i] - time[i-1]) * (
                    (threshold_arr[i-1] - coherence[i-1]) / 
                    ((coherence[i] - coherence[i-1]) - (threshold_arr[i] - threshold_arr[i-1]))
                )
                region_start = t_cross
        
        # If coherence drops below threshold and we're in a region, end the region
        elif coherence[i] < threshold_arr[i] and in_coherent_region:
            in_coherent_region = False
            region_end = time[i]
            
            # Linear interpolation to find exact end point
            if i > 0:
                t_cross = time[i-1] + (time[i] - time[i-1]) * (
                    (threshold_arr[i-1] - coherence[i-1]) / 
                    ((coherence[i] - coherence[i-1]) - (threshold_arr[i] - threshold_arr[i-1]))
                )
                region_end = t_cross
            
            coherent_regions.append((region_start, region_end))
    
    # If we're still in a coherent region at the end of the data
    if in_coherent_region:
        coherent_regions.append((region_start, time[-1]))
    
    return coherent_regions

@st.cache_data
def detect_critical_slowing(
    time: np.ndarray,
    coherence: np.ndarray,
    window_size: int = 20
) -> List[float]:
    """
    Detect critical slowing down before phase transitions.
    
    Args:
        time: Array of time points
        coherence: Array of coherence values
        window_size: Window size for calculating variance
        
    Returns:
        List of time points where critical slowing is detected
    """
    # Downsample for large datasets
    if len(coherence) > 1000:
        time, coherence = downsample_time_series(time, coherence, 1000)
    
    # Need sufficient data points for this analysis
    if len(coherence) < window_size * 2:
        return []
    
    # Calculate rolling variance
    variance = []
    for i in range(window_size, len(coherence) - window_size):
        window = coherence[i-window_size:i+window_size]
        variance.append(np.var(window))
    
    # Time points corresponding to variance calculations
    var_time = time[window_size:len(coherence)-window_size]
    
    # Detect peaks in variance (critical slowing)
    variance = np.array(variance)
    critical_points = []
    
    for i in range(1, len(variance)-1):
        # Local maximum in variance exceeding a threshold
        if (variance[i] > variance[i-1] and 
            variance[i] > variance[i+1] and 
            variance[i] > np.mean(variance) + np.std(variance)):
            critical_points.append(var_time[i])
    
    return critical_points

@st.cache_data
def plot_with_interpretations(
    time: np.ndarray,
    coherence: np.ndarray,
    entropy: np.ndarray,
    threshold: Union[np.ndarray, float],
    experiment_name: str = "Experiment"
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Create enhanced plot with interpretations for Basic Mode.
    
    Args:
        time: Array of time points
        coherence: Array of coherence values
        entropy: Array of entropy values
        threshold: Threshold values (array or single value)
        experiment_name: Name of the experiment
        
    Returns:
        Tuple of (matplotlib figure, summary dictionary)
    """
    # Validate input arrays
    try:
        # Convert inputs to numpy arrays if they aren't already
        time = np.asarray(time)
        coherence = np.asarray(coherence)
        entropy = np.asarray(entropy)
        
        # Ensure inputs have the same length
        min_len = min(len(time), len(coherence), len(entropy))
        time = time[:min_len]
        coherence = coherence[:min_len]
        entropy = entropy[:min_len]
        
        # Apply coherence non-negative constraint
        coherence = np.maximum(coherence, 0)
        
        # Apply entropy non-negative constraint
        entropy = np.maximum(entropy, 0)
        
        # For very large datasets, downsample for visualization
        if min_len > 500:
            time_ds, coherence_ds = downsample_time_series(time, coherence, 500, preserve_features=True)
            _, entropy_ds = downsample_time_series(time, entropy, 500, preserve_features=True)
            
            # Ensure threshold is properly sized
            if isinstance(threshold, np.ndarray):
                # Make sure threshold array has right length
                if len(threshold) > min_len:
                    threshold = threshold[:min_len]
                # Now downsample to match time_ds
                _, threshold_ds = downsample_time_series(time, threshold, 500)
                threshold = threshold_ds
            
            # Update the working arrays
            time = time_ds
            coherence = coherence_ds
            entropy = entropy_ds
    except Exception as e:
        # On error, create a figure with error message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error preparing visualization: {str(e)}", 
                ha='center', va='center', fontsize=12)
        ax.set_axis_off()
        return fig, {"error": str(e)}
    
    # Create figure and plot data
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set up axis labels
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(f"{experiment_name} - Enhanced Visualization")
        
        # Plot coherence and entropy
        coherence_line, = ax.plot(time, coherence, 'b-', label="Coherence", linewidth=2)
        entropy_line, = ax.plot(time, entropy, 'r-', label="Uncertainty", linewidth=1.5)
        
        # Add threshold
        if isinstance(threshold, (int, float)):
            threshold_value = threshold
            threshold_line = ax.axhline(y=threshold, color='g', linestyle='--', 
                                        label=f"Threshold (θ={threshold:.2f})")
        else:
            # For array threshold, plot the line and use mean for labeling
            threshold_value = np.mean(threshold)
            if len(threshold) != len(time):
                # Ensure threshold has same length as time for plotting
                if len(threshold) > len(time):
                    threshold = threshold[:len(time)]
                else:
                    # If threshold array is shorter, extend it
                    extended = np.ones_like(time) * threshold[-1]
                    extended[:len(threshold)] = threshold
                    threshold = extended
            
            threshold_line, = ax.plot(time, threshold, 'g--', 
                                     label=f"Threshold (θ≈{threshold_value:.2f})")
        
        # Detect phase transitions
        transitions = detect_phase_transitions(time, coherence, threshold)
        
        # Identify coherent regions
        coherent_regions = identify_coherent_regions(time, coherence, threshold)
        
        # Check for critical slowing
        critical_points = detect_critical_slowing(time, coherence)
        
        # Add interpretive elements to the plot
        
        # 1. Highlight coherent regions
        for start, end in coherent_regions:
            ax.axvspan(start, end, alpha=0.1, color='green', label='_nolegend_')
        
        # 2. Mark phase transitions
        for t in transitions:
            # Find the closest point in time array
            idx = np.abs(time - t).argmin()
            ax.axvline(x=t, linestyle=':', color='purple', alpha=0.7, label='_nolegend_')
            
            # Only add annotations if we have few transitions (avoid clutter)
            if len(transitions) <= 5:
                y_pos = coherence[idx] if idx < len(coherence) else threshold_value
                ax.annotate("Phase Transition", xy=(t, y_pos),
                            xytext=(10, 20), textcoords="offset points",
                            color='purple', fontsize=9,
                            arrowprops=dict(arrowstyle="->", color='purple', alpha=0.7))
                            
        # 3. Indicate critical slowing
        for t in critical_points:
            # Find the closest point in time array
            idx = np.abs(time - t).argmin()
            if idx < len(coherence):
                ax.plot(t, coherence[idx], 'yo', markersize=6, alpha=0.8, label='_nolegend_')
                
                # Only add annotations if we have few critical points (avoid clutter)
                if len(critical_points) <= 3:
                    ax.annotate("Critical Point", xy=(t, coherence[idx]),
                                xytext=(-20, -30), textcoords="offset points",
                                color='#CC7722', fontsize=9,
                                arrowprops=dict(arrowstyle="->", color='#CC7722', alpha=0.7))
                                
        # 4. Add region labels (only if we have coherent regions)
        if coherent_regions:
            ax.text(0.98, 0.95, "COHERENT", transform=ax.transAxes, ha='right', 
                    color='green', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
            ax.text(0.98, 0.05, "INCOHERENT", transform=ax.transAxes, ha='right', 
                    color='red', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
        
        # Add legend
        ax.legend(loc='upper left')
        
        # Tight layout for better spacing
        plt.tight_layout()
        
        # Prepare summary data
        summary = {
            "coherence_mean": float(np.mean(coherence)),
            "coherence_final": float(coherence[-1]) if len(coherence) > 0 else 0.0,
            "entropy_mean": float(np.mean(entropy)),
            "threshold": float(threshold_value),
            "phase_transitions": len(transitions),
            "transition_times": [float(t) for t in transitions],
            "coherent_regions": len(coherent_regions),
            "critical_points": len(critical_points)
        }
        
        return fig, summary
        
    except Exception as e:
        # On error, create a figure with error message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error creating visualization: {str(e)}", 
                ha='center', va='center', fontsize=12)
        ax.set_axis_off()
        return fig, {"error": str(e)}

@st.cache_data
def create_summary_card(
    time: np.ndarray,
    coherence: np.ndarray,
    entropy: np.ndarray,
    threshold: Union[np.ndarray, float, list],
    experiment_name: str = "Experiment"
) -> Dict[str, Any]:
    """
    Create a summary card with interpretations of the experiment results.
    
    Args:
        time: Array of time points
        coherence: Array of coherence values
        entropy: Array of entropy values
        threshold: Array of threshold values or single threshold value
        experiment_name: Name of the experiment
        
    Returns:
        Dictionary with summary information and interpretations
    """
    # Convert threshold to array if it's a scalar or list
    if isinstance(threshold, (int, float)):
        threshold_arr = np.ones_like(coherence) * threshold
    elif isinstance(threshold, list):
        threshold_arr = np.array(threshold)
    else:
        threshold_arr = threshold
    
    # Calculate key metrics
    coherence_mean = np.mean(coherence)
    coherence_max = np.max(coherence)
    coherence_final = coherence[-1]
    entropy_mean = np.mean(entropy)
    entropy_final = entropy[-1]
    
    # Get the final threshold value (handling both array and scalar cases)
    if isinstance(threshold_arr, np.ndarray):
        threshold_final = threshold_arr[-1]
    else:
        threshold_final = threshold_arr
    
    # Detect phase transitions and coherent regions
    phase_transitions = detect_phase_transitions(time, coherence, threshold_arr)
    coherent_regions = identify_coherent_regions(time, coherence, threshold_arr)
    
    # Calculate coherent time percentage
    total_time = time[-1] - time[0]
    coherent_time = sum(end - start for start, end in coherent_regions)
    coherent_percentage = (coherent_time / total_time) * 100 if total_time > 0 else 0
    
    # Determine stability level
    if coherent_percentage >= 80:
        stability = "Very Stable"
    elif coherent_percentage >= 60:
        stability = "Stable"
    elif coherent_percentage >= 40:
        stability = "Moderately Stable"
    elif coherent_percentage >= 20:
        stability = "Unstable"
    else:
        stability = "Very Unstable"
    
    # Determine final system state (making sure we compare scalar values)
    is_coherent = float(coherence_final) >= float(threshold_final)
    status = "Coherent" if is_coherent else "Incoherent"
    
    # Calculate correlation between entropy and coherence
    entropy_coherence_corr = np.corrcoef(entropy, coherence)[0, 1]
    
    # Generate plain language interpretation
    interpretation = ""
    
    # Start with overall assessment
    if is_coherent:
        interpretation += f"The system ended in a coherent state with {coherent_percentage:.1f}% of time spent in coherence. "
    else:
        interpretation += f"The system ended in an incoherent state with {coherent_percentage:.1f}% of time spent in coherence. "
    
    # Add detail about phase transitions
    if len(phase_transitions) == 0:
        interpretation += "No phase transitions were detected, indicating a stable system without major shifts. "
    elif len(phase_transitions) == 1:
        interpretation += "One phase transition was detected, showing that the system changed state during the experiment. "
    else:
        interpretation += f"{len(phase_transitions)} phase transitions were detected, indicating a dynamic system that alternated between coherent and incoherent states. "
    
    # Interpret entropy-coherence relationship
    if entropy_coherence_corr < -0.7:
        interpretation += "There is a strong negative relationship between uncertainty and coherence, which aligns with the theoretical prediction that higher uncertainty disrupts coherence. "
    elif entropy_coherence_corr < -0.3:
        interpretation += "There is a moderate negative relationship between uncertainty and coherence, suggesting that uncertainty does affect the system's stability. "
    elif entropy_coherence_corr < 0.3:
        interpretation += "There is a weak relationship between uncertainty and coherence, suggesting that other factors may be more influential in this system. "
    else:
        interpretation += "Interestingly, uncertainty and coherence show a positive correlation, which might indicate resilience or adaptive mechanisms in the system. "
    
    # Add recommendation based on final state
    if is_coherent and coherent_percentage < 50:
        interpretation += "The system achieved coherence despite spending most of its time in an incoherent state, suggesting it may be vulnerable to future perturbations."
    elif not is_coherent and coherent_percentage > 50:
        interpretation += "Although the system spent most of its time in a coherent state, it ended incoherently, suggesting a recent disruption or declining stability."
    
    # Format phase transitions for display
    pt_descriptions = []
    for i, t in enumerate(phase_transitions):
        idx = np.argmin(np.abs(time - t))
        direction = "coherent" if idx > 0 and coherence[idx] > coherence[idx-1] else "incoherent"
        pt_descriptions.append({
            "time": t,
            "description": f"System becomes {direction}"
        })
    
    # Create the summary card
    card = {
        'title': experiment_name,
        'metrics': {
            'average_coherence': f"{coherence_mean:.2f}",
            'max_coherence': f"{coherence_max:.2f}",
            'phase_transitions': f"{len(phase_transitions)}",
            'coherent_time': f"{coherent_percentage:.1f}%",
            'final_entropy': f"{entropy_final:.2f}",
            'correlation': f"{entropy_coherence_corr:.2f}"
        },
        'status': status,
        'stability_level': stability,
        'is_coherent': is_coherent,
        'interpretation': interpretation,
        'phase_transitions': pt_descriptions,
        'coherent_regions': coherent_regions,
        'final_coherence': coherence_final,
        'final_threshold': threshold_final,
        'mean_entropy': entropy_mean
    }
    
    return card 