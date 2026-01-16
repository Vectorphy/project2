# Macroeconomic Impact Analysis of Armed Conflicts

A comprehensive Python-based research project analyzing macroeconomic indicators across 12 target countries to assess the economic impact of armed conflicts.

## Project Overview

This project examines key macroeconomic indicators (GDP growth, inflation, unemployment, and FDI inflows) across a diverse set of countries to understand how conflicts affect economic performance. The analysis spans 20 years (2005-2025) and classifies data into three phases: Pre-War, During-War, and Post-War.

### Target Countries

- **Directly Involved in Conflicts**: Israel (ISR), Ethiopia (ETH), Vietnam (VNM)
- **Advanced Economies**: USA, Germany (DEU), South Korea (KOR), Norway (NOR)
- **Emerging Markets**: Poland (POL), Brazil (BRA), Bangladesh (BGD), Nigeria (NGA), Egypt (EGY)

## Key Features

### Data Collection

- **World Bank API**: Retrieves macro indicators (GDP growth, inflation, unemployment, FDI inflows)
- **Financial Data**: Uses Yahoo Finance and Federal Reserve data (FRED)
- **Trade Data**: UN Comtrade API for international trade analysis
- **Custom Indices**: Global Peace Report (GPR) Index for conflict intensity

### Data Processing

- **Missing Value Handling**: Linear interpolation for gaps in time-series data
- **War Phase Classification**: Automatic categorization into Pre-War, During-War, and Post-War periods
- **Data Cleaning**: Comprehensive validation and standardization

### Indicators Analyzed

| Indicator    | Code                 | Description                           |
| ------------ | -------------------- | ------------------------------------- |
| GDP Growth   | NY.GDP.MKTP.KD.ZG    | Annual GDP growth rate (%)            |
| Inflation    | FP.CPI.TOTL.ZG       | Consumer price inflation (%)          |
| Unemployment | SL.UEM.TOTL.ZS       | Unemployment rate (%)                 |
| FDI Inflow   | BX.KLT.DINV.WD.GD.ZS | Foreign direct investment as % of GDP |

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:

```bash
git clone https://github.com/Vectorphy/project2.git
cd project2
```

2. Install dependencies:

```bash
pip install -r req.txt
```

## Usage

### Running the Analysis

```python
python code.py
```

This will:

1. Connect to the World Bank API
2. Fetch macroeconomic data for all 12 countries (2005-2025)
3. Clean and process the data (handle missing values, assign war phases)
4. Generate `research_data_master.csv` for further analysis

### Output

- **research_data_master.csv**: Master dataset with all indicators, countries, years, and war phase classifications

## Dependencies

### Data Extraction & APIs

- `wbgapi` (1.0.12) - World Bank data access
- `yfinance` (0.2.36) - Financial data
- `pandas-datareader` (0.10.0) - Federal Reserve Economic Data (FRED)
- `comtradeapicall` (1.0.0) - UN Comtrade trade data
- `requests` (2.31.0) - HTTP requests for custom APIs

### Data Processing

- `pandas` (2.2.0) - DataFrames and data manipulation
- `numpy` (1.26.3) - Numerical computing
- `openpyxl` (3.1.2) - Excel file handling

### Statistical & Econometric Analysis

- `statsmodels` (0.14.1) - Regression, Difference-in-Differences, statistical tests
- `linearmodels` (5.3) - Advanced causal inference
- `pmdarima` (2.0.4) - Time-series forecasting (ARIMA)
- `scipy` (1.11.4) - Scientific computing

### Visualization

- `matplotlib` (3.8.2) - Base plotting
- `seaborn` (0.13.1) - Statistical visualization
- `plotly` (5.18.0) - Interactive plots

### Optional

- `scikit-learn` (1.4.0) - Machine learning (Random Forest, SVR for forecasting)
- `ipykernel` (6.29.0) - Jupyter Notebook support
- `tqdm` (4.66.1) - Progress bars

## Data Methodology

### Time Period

- **20-year window**: 2005-2025
- Covers pre-war, during-war, and post-war economic cycles

### Missing Data Handling

- Linear interpolation applied at country level
- Backward fill for remaining gaps

### War Phase Classification

Current placeholder logic:

- **During-War**: Years when active conflict is recorded
- **Post-War**: Years following conflict end
- **Pre-War/Control**: All other periods

_Note: Full implementation integrates UCDP (Uppsala Conflict Data Program) dates_

## Project Structure

```
project2/
├── code.py                      # Main analysis script
├── req.txt                      # Python dependencies
├── README.md                    # This file
├── research_data_master.csv     # Output: Master dataset
└── mini_project_two.pdf         # Project documentation
```

## Future Enhancements

- Integration with UCDP conflict database for precise war phase dating
- Advanced econometric modeling (Difference-in-Differences estimation)
- Machine learning-based forecasting
- Interactive dashboard development
- Expanded indicator set (trade balance, debt ratios, etc.)

## Author

**Vectorphy**

## License

This project is for academic and research purposes.

## References

- World Bank Open Data: https://data.worldbank.org/
- Yahoo Finance: https://finance.yahoo.com/
- Federal Reserve Economic Data (FRED): https://fred.stlouisfed.org/
- UN Comtrade: https://comtrade.un.org/
- Uppsala Conflict Data Program: https://www.pcr.uu.se/research/ucdp/

---

For questions or contributions, please open an issue on GitHub.
