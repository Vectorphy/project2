# Asymmetric Resilience: 12-Country Econometric Study

## Project Overview
This project performs an econometric analysis of war-induced shocks on macroeconomic indicators across 12 countries, aiming to quantify resilience differences between First World and Third World nations.

## Project Structure

| File | Role | Description |
| :--- | :--- | :--- |
| `data_pipeline.py` | ETL (Extract, Transform, Load) | Fetches data from World Bank, cleans it, assigns war status, and creates `processed_data.csv`. |
| `analysis_engine.py` | Logic | Defines the `EconometricAnalyzer` class with statistical and econometric methods. |
| `run_analysis.py` | Execution | Runs the analysis logic on the processed data and saves results to `FINAL_ANALYSIS_RESULTS.txt`. |
| `visual_analytics.py` | Visualization | Generates publication-quality charts in the `charts/` directory. |

## Quick Start

Follow these steps to reproduce the full analysis:

1.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Use `req.txt` if that is the provided filename)*

2.  **Run Data Pipeline**:
    ```bash
    python data_pipeline.py
    ```

3.  **Run Econometric Analysis**:
    ```bash
    python run_analysis.py
    ```

4.  **Generate Visual Analytics**:
    ```bash
    python visual_analytics.py
    ```

## Key Hypotheses

*   **Resilience Gap**: Third World nations experience deeper GDP dips and slower recovery rates post-shock compared to First World nations.
*   **Volatility Transfer**: Indirectly involved nations experience higher inflation volatility due to supply chain integration.
*   **Safe Haven FDI**: During global conflicts, FDI inflows shift significantly towards "Control" (uninvolved) nations.

## File Manifest

*   **`processed_data.csv`**: The final cleaned dataset.
*   **`PRELIMINARY_REPORT.md`**: Initial data audit and descriptive stats.
*   **`FINAL_ANALYSIS_RESULTS.txt`**: Detailed econometric analysis outputs.
*   **`charts/`**: Directory containing generated visualizations (`correlation_heatmap.png`, `resilience_gap.png`, `inflation_volatility.png`).
