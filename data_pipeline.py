import wbgapi as wb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
# Groups
DIRECTLY_INVOLVED = ['ISR', 'USA', 'VNM', 'ETH']
INDIRECTLY_INVOLVED = ['DEU', 'POL', 'NGA', 'EGY']
CONTROL_UNINVOLVED = ['KOR', 'NOR', 'BGD', 'BRA']

COUNTRIES = DIRECTLY_INVOLVED + INDIRECTLY_INVOLVED + CONTROL_UNINVOLVED

INDICATORS = {
    'NY.GDP.MKTP.KD.ZG': 'GDP_Growth',
    'FP.CPI.TOTL.ZG': 'Inflation',
    'BX.KLT.DINV.WD.GD.ZS': 'FDI_Inflows',
    'NE.RSB.GNFS.ZS': 'Trade_Balance_GDP_Pct'
}

# Conflict Dates
CONFLICT_DATES = {
    'ISR': [2006, 2014, 2023, 2024],
    'ETH': [2020, 2021, 2022],
    'DEU': [2022, 2023, 2024],
    'POL': [2022, 2023, 2024]
}

START_YEAR = 2000
END_YEAR = 2024

def fetch_data():
    print("Fetching data from World Bank API...")
    data_list = []
    for ind_code, ind_name in INDICATORS.items():
        try:
            d = wb.data.DataFrame(ind_code, COUNTRIES, time=range(START_YEAR, END_YEAR + 1), numericTimeKeys=True)
            d = d.reset_index().melt(id_vars=['economy'], var_name='Year', value_name=ind_name)
            d['Year'] = d['Year'].astype(int)
            d.rename(columns={'economy': 'Country'}, inplace=True)
            d.set_index(['Country', 'Year'], inplace=True)
            data_list.append(d)
        except Exception as e:
            print(f"Error fetching {ind_name}: {e}")

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
        # Include all numeric columns in stats
        cols_to_stat = list(INDICATORS.values())
        stats = df.groupby(['Country', 'War_Status'])[cols_to_stat].agg(['mean', 'median', 'std'])
        f.write(stats.to_markdown())
        f.write("\n\n")

        # 3. Visualization
        f.write("## 3. Visualization\n\n")
        f.write("![Comparative Trends](comparative_trends.png)\n")

    # Generate Plot with 3 Subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

    # Define groups and titles
    groups = [
        (DIRECTLY_INVOLVED, 'Directly Involved Countries'),
        (INDIRECTLY_INVOLVED, 'Indirectly Involved Countries'),
        (CONTROL_UNINVOLVED, 'Control/Uninvolved Countries')
    ]

    for ax, (group_countries, title) in zip(axes, groups):
        subset = df[df['Country'].isin(group_countries)]
        sns.lineplot(data=subset, x='Year', y='GDP_Growth', hue='Country', marker='o', ax=ax)
        ax.set_title(f'GDP Growth: {title}')
        ax.set_ylabel('GDP Growth (%)')
        ax.grid(True)
        # Add vertical line at 2022
        ax.axvline(x=2022, color='red', linestyle='--', linewidth=2, label='Global Shift (2022)')
        # Add legend if not present (hue adds it, but axvline might need handling or just be there)

    plt.tight_layout()
    plt.savefig('comparative_trends.png')
    plt.close()
    print("Report and chart generated.")

def main():
    # 1. API Connection
    df = fetch_data()

    # Audit before cleaning
    missing_before = df.groupby('Country')[list(INDICATORS.values())].apply(lambda x: x.isnull().sum())

    # 2. Clean & Sort
    df = df.sort_values(by=['Country', 'Year'])

    df = df.set_index(['Country', 'Year'])
    df_interpolated = df.groupby('Country').apply(lambda x: x.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill'))

    if df_interpolated.index.nlevels > 2:
        df_interpolated = df_interpolated.droplevel(0)

    df_final = df_interpolated.reset_index()

    # Audit after cleaning
    missing_after = df_final.groupby('Country')[list(INDICATORS.values())].apply(lambda x: x.isnull().sum())

    # 3. Tagging
    df_final['War_Status'] = df_final.apply(assign_war_status, axis=1)

    # 4. Automated Reporting
    generate_report(df_final, missing_before, missing_after)

    df_final.to_csv('processed_data.csv', index=False)
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
