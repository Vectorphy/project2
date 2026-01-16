import wbgapi as wb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
COUNTRIES = [
    'ISR', 'USA', 'VNM', 'ETH',  # Directly Involved
    'DEU', 'POL', 'NGA', 'EGY',  # Indirectly Involved
    'KOR', 'NOR', 'BGD', 'BRA'   # Control/Uninvolved
]

DIRECTLY_INVOLVED = ['ISR', 'USA', 'VNM', 'ETH']

INDICATORS = {
    'NY.GDP.MKTP.KD.ZG': 'GDP_Growth',
    'FP.CPI.TOTL.ZG': 'Inflation',
    'BX.KLT.DINV.WD.GD.ZS': 'FDI_Inflows'
}

# Conflict Dates based on proposal (Partial data available)
# Format: Country: [List of War Years]
CONFLICT_DATES = {
    'ISR': [2023, 2024],
    'ETH': [2020, 2021, 2022]
    # USA and VNM dates to be added based on proposal details
}

START_YEAR = 2000
END_YEAR = 2024

def fetch_data():
    print("Fetching data from World Bank API...")

    # Better approach with wbgapi for multiple indicators:
    data_list = []
    for ind_code, ind_name in INDICATORS.items():
        try:
            # Fetch data using numericTimeKeys=True so years are integers
            d = wb.data.DataFrame(ind_code, COUNTRIES, time=range(START_YEAR, END_YEAR + 1), numericTimeKeys=True)

            # Reset index to get Country as a column (wbgapi returns economy as index)
            # Melt to long format: id_vars is 'economy', rest are years
            d = d.reset_index().melt(id_vars=['economy'], var_name='Year', value_name=ind_name)

            # Convert Year to int (it might come as string or int depending on wbgapi version, but numericTimeKeys usually gives int cols.
            # However, melt converts column names to values. If columns were int, values are int.
            d['Year'] = d['Year'].astype(int)
            d.rename(columns={'economy': 'Country'}, inplace=True)

            # Set index for easy joining
            d.set_index(['Country', 'Year'], inplace=True)
            data_list.append(d)
        except Exception as e:
            print(f"Error fetching {ind_name}: {e}")

    # Combine all indicators
    if not data_list:
        raise ValueError("No data fetched.")

    final_df = pd.concat(data_list, axis=1).reset_index()
    return final_df

def assign_war_status(row):
    country = row['Country']
    year = row['Year']

    if country in CONFLICT_DATES:
        if year in CONFLICT_DATES[country]:
            return "War"

    return "Peace"

def generate_report(df, missing_before, missing_after):
    print("Generating report...")

    with open("PRELIMINARY_REPORT.md", "w") as f:
        f.write("# PRELIMINARY REPORT: Asymmetric Resilience\n\n")

        # 1. Data Quality Audit
        f.write("## 1. Data Quality Audit\n\n")
        f.write("### Missing Values Count (Before Interpolation)\n")
        f.write(missing_before.to_markdown())
        f.write("\n\n")
        f.write("### Missing Values Count (After Interpolation)\n")
        f.write(missing_after.to_markdown())
        f.write("\n\n")

        # 2. Descriptive Statistics
        f.write("## 2. Descriptive Statistics\n\n")
        stats = df.groupby(['Country', 'War_Status'])[['GDP_Growth', 'Inflation']].agg(['mean', 'median', 'std'])
        f.write(stats.to_markdown())
        f.write("\n\n")

        # 3. Visualization
        f.write("## 3. Visualization\n\n")
        f.write("![GDP Trends for Directly Involved Countries](gdp_trends.png)\n")

    # Generate Plot
    plt.figure(figsize=(12, 6))
    subset = df[df['Country'].isin(DIRECTLY_INVOLVED)]
    sns.lineplot(data=subset, x='Year', y='GDP_Growth', hue='Country', marker='o')
    plt.title('GDP Growth Trends: Directly Involved Countries (2000-2024)')
    plt.ylabel('GDP Growth (%)')
    plt.grid(True)
    plt.savefig('gdp_trends.png')
    plt.close()
    print("Report and chart generated.")

def main():
    # 1. API Connection
    df = fetch_data()

    # Audit before cleaning
    # We want to see missing values per country.
    # Group by Country and count nulls in indicator columns
    missing_before = df.groupby('Country')[list(INDICATORS.values())].apply(lambda x: x.isnull().sum())

    # 2. Clean & Sort
    # Sort
    df = df.sort_values(by=['Country', 'Year'])

    # Interpolation (Linear) - specifically handling gaps for all, ensuring ETH/NGA are covered
    # We group by country to interpolate within each country's time series
    # Using transform is cleaner than apply for keeping shape, but transform runs per column.
    # Alternatively, set index and apply
    df = df.set_index(['Country', 'Year'])

    # Interpolate each group. We use apply because we want to interpolate on the DataFrame group (all cols)
    # The result of apply on groupby object usually retains the index if the function returns a DF with same index.
    df_interpolated = df.groupby('Country').apply(lambda x: x.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill'))

    # If groupby keys are added to index, drop them.
    # Check index nlevels
    if df_interpolated.index.nlevels > 2:
        # Likely it added Country again.
        df_interpolated = df_interpolated.droplevel(0)

    df_final = df_interpolated.reset_index()

    # Audit after cleaning
    missing_after = df_final.groupby('Country')[list(INDICATORS.values())].apply(lambda x: x.isnull().sum())

    # 3. Tagging
    df_final['War_Status'] = df_final.apply(assign_war_status, axis=1)

    # 4. Automated Reporting
    generate_report(df_final, missing_before, missing_after)

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
