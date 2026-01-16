import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Configuration
DIRECTLY_INVOLVED = ['ISR', 'USA', 'VNM', 'ETH']
INDIRECTLY_INVOLVED = ['DEU', 'POL', 'NGA', 'EGY']
CONTROL_UNINVOLVED = ['KOR', 'NOR', 'BGD', 'BRA']

GROUPS = {
    'Directly Involved': DIRECTLY_INVOLVED,
    'Indirectly Involved': INDIRECTLY_INVOLVED,
    'Control': CONTROL_UNINVOLVED
}

def load_data():
    if not os.path.exists('processed_data.csv'):
        print("Error: processed_data.csv not found. Run data_pipeline.py first.")
        return None
    return pd.read_csv('processed_data.csv')

def create_charts_dir():
    if not os.path.exists('charts'):
        os.makedirs('charts')

def plot_correlation_heatmap(df):
    print("Generating Correlation Heatmap...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    indicators = ['GDP_Growth', 'Inflation', 'FDI_Inflows']

    for ax, (group_name, countries) in zip(axes, GROUPS.items()):
        subset = df[df['Country'].isin(countries)]
        if subset.empty:
            continue

        corr = subset[indicators].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax, square=True)
        ax.set_title(group_name)

    plt.tight_layout()
    plt.savefig('charts/correlation_heatmap.png')
    plt.close()

def plot_resilience_gap(df):
    print("Generating Resilience Gap Chart...")
    # Pre-Crisis (2010–2019) and Post-Crisis (2020–2024)
    pre_crisis = df[(df['Year'] >= 2010) & (df['Year'] <= 2019)].copy()
    pre_crisis['Period'] = 'Pre-Crisis (2010-2019)'

    post_crisis = df[(df['Year'] >= 2020) & (df['Year'] <= 2024)].copy()
    post_crisis['Period'] = 'Post-Crisis (2020-2024)'

    combined = pd.concat([pre_crisis, post_crisis])

    # Calculate Mean GDP per Country and Period
    # We can let seaborn do the aggregation or aggregate first. Aggregating first is safer for labels.
    agg = combined.groupby(['Country', 'Period'])['GDP_Growth'].mean().reset_index()

    plt.figure(figsize=(14, 7))
    sns.barplot(data=agg, x='Country', y='GDP_Growth', hue='Period')
    plt.title('Resilience Gap: Mean GDP Growth (Pre vs Post Crisis)')
    plt.ylabel('Mean GDP Growth (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('charts/resilience_gap.png')
    plt.close()

def plot_inflation_volatility(df):
    print("Generating Inflation Volatility Boxplot...")

    # Map countries to groups
    country_to_group = {}
    for group, countries in GROUPS.items():
        for country in countries:
            country_to_group[country] = group

    df['Group'] = df['Country'].map(country_to_group)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Group', y='Inflation', hue='Group', palette="Set2")
    plt.title('Inflation Volatility by Category')
    plt.ylabel('Inflation (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('charts/inflation_volatility.png')
    plt.close()

def main():
    df = load_data()
    if df is None:
        return

    create_charts_dir()

    plot_correlation_heatmap(df)
    plot_resilience_gap(df)
    plot_inflation_volatility(df)

    print("Visualization complete. Charts saved in 'charts/' directory.")

if __name__ == "__main__":
    main()
