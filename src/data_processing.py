import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handles merging, cleaning, imputation, and feature engineering.
    """

    def __init__(self, conflict_df: pd.DataFrame, econ_df: pd.DataFrame, metadata_df: pd.DataFrame):
        self.conflict_df = conflict_df
        self.econ_df = econ_df
        self.metadata_df = metadata_df
        self.full_df = pd.DataFrame()

    def merge_data(self) -> pd.DataFrame:
        """Merges all datasets into a single panel."""
        logger.info("Merging datasets...")

        # Merge Econ with Metadata
        # Econ has ISO3, Year
        # Metadata has ISO3
        econ_meta = pd.merge(self.econ_df, self.metadata_df, on='ISO3', how='left')

        # Merge with Conflict
        # Conflict has ISO3, Year
        # We need a left join on Econ data (base universe is World Bank countries)
        # Fill non-matches in conflict data with 0 (Peace)

        self.full_df = pd.merge(econ_meta, self.conflict_df, on=['ISO3', 'Year'], how='left')

        # Fill conflict NAs
        self.full_df['War_Binary'] = self.full_df['War_Binary'].fillna(0)
        self.full_df['Battle_Deaths'] = self.full_df['Battle_Deaths'].fillna(0)
        self.full_df['War_Intensity'] = self.full_df['War_Intensity'].fillna(0)
        self.full_df['War_Type'] = self.full_df['War_Type'].fillna('None')

        # Sort
        self.full_df.sort_values(by=['ISO3', 'Year'], inplace=True)

        return self.full_df

    def impute_missing_values(self) -> pd.DataFrame:
        """
        Applies Multiple Imputation using IterativeImputer (MICE) with Random Forest.
        """
        logger.info("Imputing missing values...")

        # Identify numeric columns for imputation
        numeric_cols = self.full_df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude ID columns if any (Year is useful for trend)
        exclude_cols = ['War_Binary', 'War_Intensity', 'Battle_Deaths'] # These are already filled 0
        cols_to_impute = [c for c in numeric_cols if c not in exclude_cols and c != 'Year']

        # We process by country group or global? Global imputation with Country Dummy is too big.
        # We can impute globally but include Income Group as feature (mapped to int).
        # Or better: Interpolate small gaps first (linear), then impute.

        # 1. Linear Interpolation for short gaps (limit 3 years)
        self.full_df[cols_to_impute] = self.full_df.groupby('ISO3')[cols_to_impute].transform(
            lambda x: x.interpolate(method='linear', limit=3, limit_direction='both')
        )

        # 2. Iterative Imputer for remaining
        # Convert Income_Group to categorical code for imputation
        self.full_df['Income_Code'] = self.full_df['Income_Group'].astype('category').cat.codes

        impute_vars = cols_to_impute + ['Year', 'Income_Code', 'War_Binary']

        # Subsetting data for imputation
        impute_data = self.full_df[impute_vars].copy()

        # Using Random Forest as estimator for robust non-linear imputation
        imp = IterativeImputer(estimator=RandomForestRegressor(n_jobs=-1, max_depth=10), max_iter=5, random_state=42)

        imputed_values = imp.fit_transform(impute_data)
        imputed_df = pd.DataFrame(imputed_values, columns=impute_vars, index=impute_data.index)

        # Update original df
        self.full_df[cols_to_impute] = imputed_df[cols_to_impute]

        # Drop helper
        self.full_df.drop(columns=['Income_Code'], inplace=True)

        return self.full_df

    def engineer_features(self) -> pd.DataFrame:
        """
        Creates derived features: Lags, Rolling Stats, Cumulative Exposure, Log-transforms.
        """
        logger.info("Engineering features...")

        # Ensure sorting
        self.full_df.sort_values(by=['ISO3', 'Year'], inplace=True)

        # 1. Log Transforms (handle zeros)
        # Battle Deaths already processed? 'Battle_Deaths' -> 'War_Intensity' was log1p in ingestion.
        # Log Inflation (can be negative? CPI growth usually positive but can be deflation.
        # Using Symmetrical Log or just raw for percentages often better.
        # GDP Per Capita is log-normal.
        self.full_df['Log_GDP_PC'] = np.log1p(self.full_df['GDP_Per_Capita_Constant'])

        # 2. Lags (t-1 to t-3)
        lags = [1, 2, 3]
        lag_cols = ['War_Binary', 'War_Intensity', 'GDP_Growth', 'Inflation']

        for col in lag_cols:
            for l in lags:
                self.full_df[f'{col}_lag{l}'] = self.full_df.groupby('ISO3')[col].shift(l)

        # 3. Cumulative Conflict
        # Rolling sum of War_Binary over past 10 years
        self.full_df['Cumulative_Conflict_10y'] = self.full_df.groupby('ISO3')['War_Binary'].transform(
            lambda x: x.rolling(window=10, min_periods=1).sum()
        )

        # 4. Volatility (Standard Deviation)
        # 5-year rolling std of Inflation and Growth
        self.full_df['Inflation_Vol_5y'] = self.full_df.groupby('ISO3')['Inflation'].transform(
            lambda x: x.rolling(window=5, min_periods=3).std()
        )
        self.full_df['Growth_Vol_5y'] = self.full_df.groupby('ISO3')['GDP_Growth'].transform(
            lambda x: x.rolling(window=5, min_periods=3).std()
        )

        # 5. Recovery Flags
        # Post-Conflict: If t-1 was War and t is Peace.
        # We need to detect switch.
        self.full_df['War_Prev'] = self.full_df.groupby('ISO3')['War_Binary'].shift(1).fillna(0)
        self.full_df['Post_Conflict_Onset'] = ((self.full_df['War_Binary'] == 0) & (self.full_df['War_Prev'] == 1)).astype(int)

        # Years since conflict end
        # This is complex in vectorized pandas.
        # Simplified: If Post_Conflict_Onset, counter starts?
        # We will skip complex counter for now, use War_Binary lags.

        # Drop initial rows with NaNs from lags
        # Actually keep them but ML models might need dropna.
        # We will keep them for now.

        return self.full_df

    def get_clean_panel(self) -> pd.DataFrame:
        return self.full_df

if __name__ == "__main__":
    # Test stub
    pass
