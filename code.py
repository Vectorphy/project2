import pandas as pd
import wbgapi as wb
import datetime

# 1. Configuration: Your 12 Target Countries
countries = ['ISR', 'USA', 'DEU', 'POL', 'KOR',
             'NOR', 'VNM', 'ETH', 'NGA', 'EGY', 'BGD', 'BRA']

# [cite_start]Indicators from your proposal [cite: 113-116]
indicators = {
    'NY.GDP.MKTP.KD.ZG': 'GDP_Growth',
    'FP.CPI.TOTL.ZG': 'Inflation',
    'SL.UEM.TOTL.ZS': 'Unemployment',
    'BX.KLT.DINV.WD.GD.ZS': 'FDI_Inflow_GDP_Pct'
}


def get_macro_data(country_list, indicator_map):
    print("Connecting to World Bank API...")
    # [cite_start]Fetch data for the last 20 years to cover Pre/During/Post war cycles [cite: 182]
    df = wb.data.DataFrame(indicator_map.keys(), country_list, time=range(
        2005, 2025), numericTimeKeys=True)

    # Reshape and clean
    df = df.stack().unstack(level=1)
    df.index.names = ['Country', 'Year']
    df.columns = [indicator_map[col] for col in df.columns]
    return df.reset_index()


def clean_and_sort(df):
    print("Cleaning and Sorting Data...")

    # [cite_start]1. Handle Missing Values (Linear Interpolation for Third World gaps) [cite: 107]
    df = df.groupby('Country').apply(lambda x: x.interpolate(
        method='linear').fillna(method='bfill'))

    # [cite_start]2. Add 'War Phase' Logic [cite: 122-124]
    # Note: In your full project, you will merge this with UCDP dates
    # This is a placeholder logic for the 'Directly Involved' group
    conflict_years = {
        'ISR': [2023, 2024],
        'ETH': [2020, 2021, 2022],
        'VNM': [1975]  # Historical comparison
    }

    def assign_phase(row):
        if row['Country'] in conflict_years and row['Year'] in conflict_years[row['Country']]:
            return 'During-War'
        elif row['Country'] in conflict_years and row['Year'] > max(conflict_years[row['Country']]):
            return 'Post-War'
        else:
            return 'Pre-War/Control'

    df['War_Phase'] = df.apply(assign_phase, axis=1)

    # 3. Sorting for Analysis
    df = df.sort_values(by=['Country', 'Year'])
    return df


# Execution
raw_data = get_macro_data(countries, indicators)
clean_data = clean_and_sort(raw_data)

# Save for your LateX project
clean_data.to_csv('research_data_master.csv', index=False)
print("Success! 'research_data_master.csv' is ready for your analysis.")
