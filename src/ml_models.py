import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import partial_dependence
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConflictML:
    """
    Machine Learning Analysis: Prediction and Clustering of Post-Conflict Recovery.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def prepare_data(self):
        """Prepares data for ML (Drop NaNs, Encode categoricals)."""
        # Sort by Year
        if 'Year' in self.df.columns:
            self.df.sort_values('Year', inplace=True)

        # Select Features
        # Exclude metadata like ISO3 unless encoded?
        # We will drop ISO3 for the model to generalize patterns, not memorize countries.
        # But Income Group is important.

        feature_cols = [
            'War_Binary', 'War_Intensity', 'Cumulative_Conflict_10y',
            'GDP_Growth_lag1', 'GDP_Growth_lag2', 'Inflation_lag1',
            'Trade_Openness', 'Govt_Expenditure_GDP', 'Log_GDP_PC',
            'Inflation_Vol_5y', 'Growth_Vol_5y'
        ]

        # Add Income Code
        if 'Income_Group' in self.df.columns:
            self.df['Income_Code'] = self.df['Income_Group'].astype('category').cat.codes
            feature_cols.append('Income_Code')

        target_col = 'GDP_Growth'

        # Drop rows with missing values
        ml_data = self.df[feature_cols + [target_col, 'Year', 'ISO3']].dropna()

        return ml_data, feature_cols, target_col

    def train_xgboost(self):
        """
        Trains XGBoost with Time Series Cross-Validation.
        """
        logger.info("Training XGBoost Model...")

        ml_data, feature_cols, target_col = self.prepare_data()

        X = ml_data[feature_cols]
        y = ml_data[target_col]

        tscv = TimeSeriesSplit(n_splits=5)

        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, random_state=42)

        scores = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            scores.append(rmse)

        logger.info(f"CV RMSE Scores: {scores}")
        mean_rmse = np.mean(scores)
        logger.info(f"Average RMSE: {mean_rmse}")

        # Fit on full data for feature importance
        model.fit(X, y)

        return model, feature_cols, mean_rmse

    def train_random_forest(self):
        """
        Trains Random Forest for comparison.
        """
        logger.info("Training Random Forest Model...")

        ml_data, feature_cols, target_col = self.prepare_data()

        X = ml_data[feature_cols]
        y = ml_data[target_col]

        # No CV here for brevity, just fit
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X, y)

        return model

    def cluster_recovery_trajectories(self):
        """
        Identifies clusters of post-conflict growth trajectories (t to t+4).
        """
        logger.info("Clustering Recovery Trajectories...")

        # 1. Identify Conflict End Points (Transition War -> Peace)
        # Shift War Status
        self.df.sort_values(['ISO3', 'Year'], inplace=True)
        self.df['War_Next'] = self.df.groupby('ISO3')['War_Binary'].shift(-1)

        # End of war: Current is War (1), Next is Peace (0).
        # Actually, recovery starts the year AFTER war ends.
        # So if Year T is War, and T+1 is Peace. T+1 is year 1 of recovery.

        # We find Year T where War=1 and War_Next=0.
        conflict_ends = self.df[(self.df['War_Binary'] == 1) & (self.df['War_Next'] == 0)].copy()

        trajectories = []
        labels = []

        for idx, row in conflict_ends.iterrows():
            iso = row['ISO3']
            end_year = row['Year']

            # We want growth for [end_year+1, ..., end_year+5]
            # Verify data availability

            # Get country data
            c_data = self.df[self.df['ISO3'] == iso].set_index('Year')

            years_needed = range(int(end_year)+1, int(end_year)+6)
            if all(y in c_data.index for y in years_needed):
                 growth_path = c_data.loc[years_needed, 'GDP_Growth'].values
                 trajectories.append(growth_path)
                 labels.append(f"{iso}_{end_year}")

        if not trajectories:
            logger.warning("No complete recovery trajectories found (5 years post-war).")
            return None, None

        X_traj = np.array(trajectories)

        # Normalize? Growth rates are comparable.

        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_traj)

        # Return dataframe with labels and cluster
        result = pd.DataFrame({
            'ID': labels,
            'Cluster': clusters,
            'Trajectory': list(X_traj)
        })

        return result, kmeans

if __name__ == "__main__":
    pass
