# Asymmetric Resilience: 12-Country Econometric Study

## Project Overview
This project performs an econometric analysis of war-induced shocks on macroeconomic indicators across 12 countries, classified into "Directly Involved", "Indirectly Involved", and "Control" groups. It aims to quantify resilience differences between First World and Third World nations.

## Setup

1.  **Clone the repository**.
2.  **Install Dependencies**:
    Ensure you have Python 3.8+ installed. Install the required packages using:
    ```bash
    pip install -r req.txt
    pip install tabulate
    ```
    *Note: `req.txt` contains the core dependencies. `tabulate` is required for Markdown table generation.*

## Execution

### 1. Data Pipeline
To fetch data from the World Bank API, clean it, and generate the preliminary report and visualization, run:

```bash
python data_pipeline.py
```

This will create:
*   `PRELIMINARY_REPORT.md`: A report containing data audits and descriptive statistics.
*   `comparative_trends.png`: A visualization of GDP growth trends across the three country groups.

### 2. Econometric Analysis
The `analysis_engine.py` module contains the `EconometricAnalyzer` class for modular analysis. You can use it in a Python script or Jupyter Notebook as follows:

```python
import pandas as pd
from analysis_engine import EconometricAnalyzer

# Assuming you have the processed dataframe 'df' (e.g., from data_pipeline execution or CSV)
# df = pd.read_csv('your_data.csv')

analyzer = EconometricAnalyzer()

# 1. Compare Volatility
volatility = analyzer.compare_volatility(df)
print(volatility)

# 2. Calculate Resilience Gap
gap = analyzer.calculate_resilience_gap(df)
print(gap)

# 3. Correlation Matrix
correlations = analyzer.correlation_matrix(df)
print(correlations)
```

## File Manifest

*   **`data_pipeline.py`**: The main script that connects to the World Bank API, fetches economic indicators (GDP Growth, Inflation, FDI Inflows, Trade Balance), cleans the data (interpolating missing values), assigns War/Peace status, and generates the preliminary report and plots.
*   **`analysis_engine.py`**: A modular Python file containing the `EconometricAnalyzer` class. It provides methods for specific econometric calculations: volatility comparison, resilience gap analysis, and correlation matrices.
*   **`PRELIMINARY_REPORT.md`**: An automatically generated Markdown report (by `data_pipeline.py`) that presents a data quality audit, descriptive statistics, and embeds the visualization.
*   **`comparative_trends.png`**: A generated image showing GDP growth trends for the three country groups with a marker for the 2022 global shift.
