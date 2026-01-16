import pandas as pd
import numpy as np

# Define Groups
DIRECTLY_INVOLVED = ['ISR', 'USA', 'VNM', 'ETH']
INDIRECTLY_INVOLVED = ['DEU', 'POL', 'NGA', 'EGY']
CONTROL_UNINVOLVED = ['KOR', 'NOR', 'BGD', 'BRA']

GROUPS = {
    'Directly Involved': DIRECTLY_INVOLVED,
    'Indirectly Involved': INDIRECTLY_INVOLVED,
    'Control': CONTROL_UNINVOLVED
}

def main():
    print("Loading data...")
    try:
        df = pd.read_csv('processed_data.csv')
    except FileNotFoundError:
        print("Error: processed_data.csv not found. Please run data_pipeline.py first.")
        return

    # --- Metric Engineering ---
    print("Engineering metrics...")

    # 1. Volatility (Std Dev of Inflation per Country)
    # Calculate std deviation of Inflation for each country across the entire timeframe
    volatility = df.groupby('Country')['Inflation'].std().reset_index()
    volatility.rename(columns={'Inflation': 'Metric_Volatility'}, inplace=True)

    # Merge back to main dataframe
    df = pd.merge(df, volatility, on='Country', how='left')

    # 2. Resilience Gap (Post-2020 avg GDP - Pre-2020 avg GDP)
    # Pre-2020: 2010-2019
    # Post-2020: 2020-2024

    pre_period = df[(df['Year'] >= 2010) & (df['Year'] <= 2019)]
    post_period = df[(df['Year'] >= 2020) & (df['Year'] <= 2024)]

    avg_pre = pre_period.groupby('Country')['GDP_Growth'].mean()
    avg_post = post_period.groupby('Country')['GDP_Growth'].mean()

    resilience_gap = avg_post - avg_pre
    resilience_gap_df = resilience_gap.reset_index()
    resilience_gap_df.rename(columns={'GDP_Growth': 'Metric_Resilience_Gap'}, inplace=True)

    # Merge back to main dataframe
    df = pd.merge(df, resilience_gap_df, on='Country', how='left')

    # Assign Groups
    country_to_group = {}
    for group_name, countries in GROUPS.items():
        for country in countries:
            country_to_group[country] = group_name

    df['Group'] = df['Country'].map(country_to_group)

    # Save tableau_master.csv
    print("Saving tableau_master.csv...")
    df.to_csv('tableau_master.csv', index=False)

    # --- Correlation Reshaping ---
    print("Calculating correlations...")

    # Identify numeric columns for correlation
    # Note: 'Year' is numeric but usually excluded from correlation matrix in this context unless specified.
    # The prompt says "Measure_X | Measure_Y".
    # Based on existing analysis_engine.py logic, the relevant measures are likely the economic indicators.
    numeric_cols = ['GDP_Growth', 'Inflation', 'FDI_Inflows', 'Trade_Balance_GDP_Pct']

    # Verify columns exist
    missing_cols = [col for col in numeric_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for correlation: {missing_cols}")

    correlation_frames = []

    for group_name, countries in GROUPS.items():
        subset = df[df['Country'].isin(countries)]

        if subset.empty:
            continue

        # Calculate correlation matrix
        # select only numeric cols for correlation
        corr_matrix = subset[numeric_cols].corr()

        # Melt the matrix
        # Reset index moves the index (which is Measure_X) to a column
        corr_reset = corr_matrix.reset_index()
        corr_reset.rename(columns={'index': 'Measure_X'}, inplace=True) # Rename 'index' to 'Measure_X' if it was named 'index' or if it was the index name

        # If the index didn't have a name, reset_index names it 'index'.
        # If it had a name, it uses that name. The index of corr() usually has no name or index name.

        corr_melted = corr_reset.melt(id_vars='Measure_X', var_name='Measure_Y', value_name='Correlation_Value')

        corr_melted['Group'] = group_name

        # Reorder columns
        corr_melted = corr_melted[['Group', 'Measure_X', 'Measure_Y', 'Correlation_Value']]

        correlation_frames.append(corr_melted)

    if correlation_frames:
        final_correlations = pd.concat(correlation_frames, ignore_index=True)
        # Save tableau_correlations.csv
        print("Saving tableau_correlations.csv...")
        final_correlations.to_csv('tableau_correlations.csv', index=False)
    else:
        print("No correlations calculated.")

    print("Tableau preparation completed.")

if __name__ == "__main__":
    main()
