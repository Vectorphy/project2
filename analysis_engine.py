import pandas as pd
import numpy as np

class EconometricAnalyzer:
    def __init__(self):
        pass

    def compare_volatility(self, df):
        """
        Calculate and return the Standard Deviation (std) of Inflation and GDP Growth
        for "War" vs. "Peace" rows, grouped by Country.
        """
        # Ensure correct columns exist
        cols = ['Inflation', 'GDP_Growth']
        for col in cols:
            if col not in df.columns:
                raise ValueError(f"DataFrame missing required column: {col}")

        # Group by Country and War_Status, calculate std for Inflation and GDP_Growth
        volatility = df.groupby(['Country', 'War_Status'])[cols].std()
        return volatility

    def calculate_resilience_gap(self, df):
        """
        Calculate the "Dip": (Average GDP 2020-2024) - (Average GDP 2010-2019).
        Return a dataframe sorted by the magnitude of this drop.
        """
        if 'GDP_Growth' not in df.columns or 'Year' not in df.columns:
            raise ValueError("DataFrame missing 'GDP_Growth' or 'Year' columns")

        # Filter periods
        period_pre = df[(df['Year'] >= 2010) & (df['Year'] <= 2019)]
        period_post = df[(df['Year'] >= 2020) & (df['Year'] <= 2024)]

        # Calculate averages per country
        avg_pre = period_pre.groupby('Country')['GDP_Growth'].mean()
        avg_post = period_post.groupby('Country')['GDP_Growth'].mean()

        # Calculate Dip (Post - Pre)
        resilience_gap = avg_post - avg_pre

        # Sort by magnitude of the drop (ascending, so largest drops are first)
        resilience_gap = resilience_gap.sort_values(ascending=True)

        return resilience_gap.to_frame(name='Resilience_Gap')

    def correlation_matrix(self, df):
        """
        Compute the correlation between Inflation and FDI_Inflows for each of the
        three groups (Directly Involved, Indirectly Involved, Control).
        """
        if 'Inflation' not in df.columns or 'FDI_Inflows' not in df.columns:
             raise ValueError("DataFrame missing 'Inflation' or 'FDI_Inflows' columns")

        # Define groups
        DIRECTLY_INVOLVED = ['ISR', 'USA', 'VNM', 'ETH']
        INDIRECTLY_INVOLVED = ['DEU', 'POL', 'NGA', 'EGY']
        CONTROL_UNINVOLVED = ['KOR', 'NOR', 'BGD', 'BRA']

        groups = {
            'Directly Involved': DIRECTLY_INVOLVED,
            'Indirectly Involved': INDIRECTLY_INVOLVED,
            'Control': CONTROL_UNINVOLVED
        }

        correlations = {}

        for name, countries in groups.items():
            subset = df[df['Country'].isin(countries)]
            if not subset.empty:
                # Calculate correlation
                corr = subset['Inflation'].corr(subset['FDI_Inflows'])
                correlations[name] = corr
            else:
                correlations[name] = np.nan

        return pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation_Inflation_FDI'])
