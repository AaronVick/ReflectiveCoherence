# Data Export Guide

## Introduction

The Reflective Coherence Explorer allows you to export your experiment data for further analysis, sharing, or archiving. This guide explains how to use the data export features, what data formats are available, and how to work with the exported data.

## For General Users: Simple Guide

### Exporting Your Experiment Data

1. **From the Dashboard**: After running an experiment, look for the "Export" button in the results section.

2. **Export Options**: You can export your data in several formats:
   - CSV file (for spreadsheet programs like Excel)
   - JSON file (for data analysis programs)
   - Image files of visualizations (PNG format)

3. **Accessing Exported Files**: All exported files are saved to the `data/exports` folder in your Reflective Coherence installation directory.

4. **Quick Analysis Tips**:
   - Use Excel or Google Sheets to open CSV files and create custom charts
   - Share PNG images directly in presentations or reports
   - Use the JSON files to load your experiment into the application again later

### What's Included in Exports

Each export includes:
- Parameter values (α, β, K, etc.)
- Time series data (coherence and entropy values)
- Calculated threshold values
- Metadata (experiment ID, description, date/time)

## For Scientists and Researchers: Advanced Guide

### Data Export Architecture

The Reflective Coherence Explorer uses a structured data export system that preserves all experiment parameters, results, and metadata. Exports are generated directly from the internal data model used in simulations, ensuring fidelity between what you see in the application and the exported data.

### Export Formats and Specifications

#### CSV Format
```
time,coherence,entropy
0.0,0.5,0.3
0.401606425702811,0.5020261991260422,0.3
0.8032128514056219,0.5040726397961426,0.3
...
```

The CSV export uses a standard comma-separated format with headers. It includes:
- Primary time series data (time, coherence, entropy) with maximum precision
- One row per simulation timestep

#### JSON Format
```json
{
  "experiment_id": "exp_01",
  "description": "Basic coherence accumulation experiment",
  "parameters": {
    "alpha": 0.1,
    "K": 1.0,
    "beta": 0.2,
    "initial_coherence": 0.5,
    "time_range": [0.0, 100.0],
    "time_steps": 500
  },
  "results": {
    "threshold": 0.3,
    "final_coherence": 0.7532,
    "mean_entropy": 0.3,
    "entropy_variance": 0.0
  },
  "time_series": {
    "time": [0.0, 0.401606425702811, ...],
    "coherence": [0.5, 0.5020261991260422, ...],
    "entropy": [0.3, 0.3, ...]
  },
  "created_at": "2023-10-15T14:30:22.123456",
  "run_at": "2023-10-15T14:30:25.654321"
}
```

The JSON export includes:
- Complete experiment metadata and parameters
- Summary statistics (threshold, final values, means, variances)
- Full time series data with original precision
- Timestamps for experiment creation and execution

### Programmatic Access to Exported Data

For advanced analysis, you can directly access the export files programmatically:

```python
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Load experiment from CSV
df = pd.read_csv('data/exports/exp_01_20231015_143025.csv')

# Calculate additional metrics
df['coherence_velocity'] = df['coherence'].diff() / df['time'].diff()
df['coherence_acceleration'] = df['coherence_velocity'].diff() / df['time'].diff()

# Find phase transition points
threshold = 0.3  # From experiment
phase_transitions = df[np.abs(df['coherence'] - threshold) < 0.01].index.tolist()

# Plot advanced visualization
plt.figure(figsize=(12, 8))
plt.plot(df['time'], df['coherence'], 'b-', label='Coherence')
plt.plot(df['time'], df['entropy'], 'r-', label='Entropy')
plt.plot(df['time'], df['coherence_velocity'], 'g--', label='dC/dt')
plt.axhline(y=threshold, color='k', linestyle=':', label='Threshold')
for pt in phase_transitions:
    plt.axvline(x=df.iloc[pt]['time'], color='m', linestyle='--')
plt.legend()
plt.title('Advanced Coherence Analysis')
plt.savefig('advanced_analysis.png', dpi=300)
```

### Statistical Analysis Guidance

Exported data is suitable for various statistical analyses:

1. **Time Series Analysis**:
   - ARIMA modeling for forecasting coherence trends
   - Autocorrelation analysis to identify cyclic patterns
   - Granger causality tests between entropy and coherence

2. **Comparative Statistics**:
   - ANOVA for comparing outcomes across parameter sets
   - Multi-factor regression to identify parameter interactions
   - Statistical power analysis for experiment design

3. **Phase Transition Analysis**:
   - Lyapunov exponent calculation for stability analysis
   - Bifurcation diagrams for parameter exploration
   - Critical slowing down detection near transition points

### Integration with Scientific Computing Ecosystems

The export formats are compatible with common scientific computing tools:

- **Python**: Compatible with NumPy, Pandas, SciPy, scikit-learn, and TensorFlow
- **R**: Easily imported using read.csv() or jsonlite package
- **MATLAB**: Direct import using csvread() or jsondecode()
- **Julia**: Supported via CSV.jl and JSON.jl packages

### Batch Processing of Exports

For large-scale analysis, you can use the built-in batch processing capabilities:

```bash
# From the application root directory
python tools/batch_analysis.py --input data/exports --pattern "exp_*_20231015_*.json" --output analysis_results
```

This will run a standard suite of analyses across all matching experiment files and generate a comparative report.

## Troubleshooting

### Common Issues

1. **Missing Export Button**: Make sure you've completed running an experiment before trying to export.

2. **Export Fails**: Check that the `data/exports` directory exists and is writable.

3. **Large File Warnings**: Very long simulations may generate large export files. Consider reducing time steps for more manageable exports.

### Getting Help

For additional assistance with data exports, see the [User Guide](USER_GUIDE.md) or submit an issue on our GitHub repository. 