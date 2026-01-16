# Asymmetric Resilience: 12-Country Econometric Study

## Repository Structure

| File | Role | Description |
| :--- | :--- | :--- |
| `data_pipeline.py` | Data ETL | Fetches data from World Bank, cleans it, assigns war status, and creates `processed_data.csv`. |
| `analysis_engine.py` | Econometrics | Defines the `EconometricAnalyzer` class with statistical and econometric methods. |
| `run_analysis.py` | Execution | Runs the analysis logic on the processed data and saves results to `FINAL_ANALYSIS_RESULTS.txt`. |
| `visual_analytics.py` | Visualization | Generates publication-quality charts in the `charts/` directory. |

## Execution Guide

Follow these steps to run the full pipeline:

1.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Use `req.txt` if that is the provided filename)*

2.  **Step 1: Data Pipeline**:
    ```bash
    python data_pipeline.py
    ```

3.  **Step 2: Econometric Analysis**:
    ```bash
    python run_analysis.py
    ```

4.  **Step 3: Visualization**:
    ```bash
    python visual_analytics.py
    ```

## Output Manifest

The results of the analysis can be found in the following locations:

*   **`charts/`**: Directory containing the generated visualizations:
    *   `correlation_heatmap.png`
    *   `resilience_gap.png`
    *   `inflation_volatility.png`
*   **`FINAL_ANALYSIS_RESULTS.txt`**: Detailed text report containing volatility analysis, resilience gap calculations, and correlation matrices.
*   **`processed_data.csv`**: The clean dataset used for analysis.
