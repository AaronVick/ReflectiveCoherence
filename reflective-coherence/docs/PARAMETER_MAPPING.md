# Parameter Mapping Documentation

This document maintains a precise mapping between the simplified UI elements in Basic Mode and their corresponding mathematical parameters in the underlying model. This ensures complete scientific integrity while providing a more accessible interface.

## Core Parameter Mappings

| Basic Mode Name | Expert Mode Parameter | Mathematical Definition | Range Mapping | Transformation |
|----------------|---------------------|------------------------|--------------|---------------|
| System Growth Speed | α (alpha) | Coherence growth rate | Low: 0.01-0.05<br>Medium: 0.1-0.15<br>High: 0.2-0.3 | Direct 1:1 mapping |
| Uncertainty Impact | β (beta) | Entropy influence factor | Low: 0.05-0.15<br>Medium: 0.2-0.3<br>High: 0.4-0.5 | Direct 1:1 mapping |
| System Capacity | K | Maximum coherence | Small: 0.5-0.8<br>Medium: 1.0<br>Large: 1.2-2.0 | Direct 1:1 mapping |
| Starting Stability | Initial coherence (C₀) | Initial coherence value | Low: 0.1-0.3<br>Medium: 0.4-0.6<br>High: 0.7-0.9 | Direct 1:1 mapping |

## Entropy Function Mappings

| Basic Mode Name | Expert Mode Function | Mathematical Implementation | Parameter Mapping |
|----------------|---------------------|---------------------------|-----------------|
| Steady Environment | Constant Entropy | `H(t) = c` | Steady Level → c value:<br>Low: 0.1<br>Medium: 0.3<br>High: 0.5 |
| Increasing Complexity | Increasing Entropy | `H(t) = H₀ + rt` | Complexity Growth → r value:<br>Slow: 0.001<br>Medium: 0.003<br>Fast: 0.005 |
| Learning Process | Decreasing Entropy | `H(t) = H₀ - rt` | Learning Rate → r value:<br>Slow: 0.001<br>Medium: 0.003<br>Fast: 0.005 |
| Cyclical Challenges | Oscillating Entropy | `H(t) = b + a·sin(ft)` | Challenge Frequency → f value:<br>Low: 0.05<br>Medium: 0.1<br>High: 0.2<br><br>Challenge Intensity → a value:<br>Mild: 0.1<br>Moderate: 0.2<br>Severe: 0.3 |
| Random Environment | Random Entropy | `H(t) = mean + std·rand()` | Uncertainty Level → mean value:<br>Low: 0.2<br>Medium: 0.3<br>High: 0.4<br><br>Variability → std value:<br>Low: 0.05<br>Medium: 0.1<br>High: 0.2 |

## Experiment Template Parameter Sets

Each template in Basic Mode uses specific parameter sets that have been carefully validated against the mathematical model:

### Adaptation Test

**Purpose**: Tests system ability to maintain coherence as entropy gradually increases

| Parameter | Value | Justification |
|-----------|-------|---------------|
| α (alpha) | 0.15 | Moderate growth rate allows adaptation without being too responsive |
| β (beta) | 0.3 | Moderate entropy influence shows clear effects without overwhelming system |
| K | 1.0 | Standard capacity allows for standardized comparison |
| Initial Coherence | 0.6 | Starting moderately coherent to observe adaptation process |
| Entropy Function | Increasing | `H(t) = 0.1 + 0.002·t` allows gradual increase to test adaptation |
| Simulation Time | 300 | Sufficient time to observe full adaptation response |

### Resilience Study

**Purpose**: Tests system recovery after entropy disturbances

| Parameter | Value | Justification |
|-----------|-------|---------------|
| α (alpha) | 0.2 | Higher growth rate enables observation of recovery capabilities |
| β (beta) | 0.25 | Moderate entropy influence shows clear impacts from disturbances |
| K | 1.0 | Standard capacity allows for standardized comparison |
| Initial Coherence | 0.7 | Starting highly coherent to observe impact of disturbances |
| Entropy Function | Custom | `H(t) = 0.2 + impulse(t)` with entropy spikes at t=50, t=150 |
| Simulation Time | 250 | Sufficient time to observe multiple recovery cycles |

### Stability Analysis

**Purpose**: Examines steady-state behavior under constant conditions

| Parameter | Value | Justification |
|-----------|-------|---------------|
| α (alpha) | 0.1 | Standard growth rate for baseline behavior |
| β (beta) | 0.2 | Standard entropy influence for baseline behavior |
| K | 1.0 | Standard capacity allows for standardized comparison |
| Initial Coherence | 0.5 | Starting at mid-point to observe natural equilibrium |
| Entropy Function | Constant | `H(t) = 0.3` provides steady environmental conditions |
| Simulation Time | 200 | Sufficient time to reach steady state |

### Phase Transition Explorer

**Purpose**: Demonstrates transitions between coherent and incoherent states

| Parameter | Value | Justification |
|-----------|-------|---------------|
| α (alpha) | 0.12 | Moderate growth allows clear observation of transitions |
| β (beta) | 0.4 | Higher entropy influence makes transitions more pronounced |
| K | 1.0 | Standard capacity allows for standardized comparison |
| Initial Coherence | 0.5 | Starting at mid-point gives flexibility to move either direction |
| Entropy Function | Oscillating | `H(t) = 0.3 + 0.2·sin(0.05·t)` creates cycles crossing threshold |
| Simulation Time | 400 | Sufficient time to observe multiple phase transitions |

## Implementation Notes

1. **Backend Processing**: All Basic Mode UI elements map deterministically to exact scientific parameters. No approximations or simplifications occur in the mathematical model.

2. **Parameter Validation**: Before execution, all parameter combinations are validated against mathematical constraints to ensure they produce valid results.

3. **Precision Handling**: While the UI may display simplified terms, the underlying calculations use full numerical precision.

4. **Version Tracking**: This mapping document is version-controlled and will be updated whenever:
   - New simplified UI elements are added
   - Parameter ranges are modified
   - New experiment templates are created

## Verification Process

To ensure mathematical integrity:

1. Automated testing compares results between Basic Mode and direct Expert Mode parameter entry
2. All templates undergo mathematical verification to confirm expected behaviors
3. Parameter mapping code is isolated and unit tested
4. Result consistency is verified across interface modes

Current verification status: Pending initial implementation

## Revision History

| Version | Date | Changes | Verification Status |
|---------|------|---------|---------------------|
| 0.1 | YYYY-MM-DD | Initial mapping documentation | Pending | 