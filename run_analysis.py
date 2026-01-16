import pandas as pd
from analysis_engine import EconometricAnalyzer

def main():
    # Load processed data
    print("Loading processed data...")
    try:
        df = pd.read_csv('processed_data.csv')
    except FileNotFoundError:
        print("Error: 'processed_data.csv' not found. Please run data_pipeline.py first.")
        return

    # Instantiate Analyzer
    analyzer = EconometricAnalyzer()

    # Run Analyses
    print("Running econometric analyses...")
    volatility = analyzer.compare_volatility(df)
    resilience_gap = analyzer.calculate_resilience_gap(df)
    correlations = analyzer.correlation_matrix(df)

    # Save Results
    print("Saving results to FINAL_ANALYSIS_RESULTS.txt...")
    with open('FINAL_ANALYSIS_RESULTS.txt', 'w') as f:
        f.write("=== Volatility Analysis ===\n")
        f.write("(Standard Deviation of Inflation and GDP Growth for 'War' vs. 'Peace')\n\n")
        try:
             f.write(volatility.to_markdown())
        except ImportError:
             f.write(volatility.to_string())
        f.write("\n\n")

        f.write("=== Resilience Gap Analysis ===\n")
        f.write("(Drop in Average GDP: 2020-2024 vs 2010-2019)\n\n")
        try:
             f.write(resilience_gap.to_markdown())
        except ImportError:
             f.write(resilience_gap.to_string())
        f.write("\n\n")

        f.write("=== Correlation Matrix ===\n")
        f.write("(Correlation between Inflation and FDI_Inflows)\n\n")
        try:
             f.write(correlations.to_markdown())
        except ImportError:
             f.write(correlations.to_string())
        f.write("\n")

    print("Analysis complete. Results saved.")

if __name__ == "__main__":
    main()
