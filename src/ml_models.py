import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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
    Machine Learning Analysis: Prediction, Clustering, and Counterfactuals.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def prepare_data(self):
        """Prepares data for ML (Drop NaNs, Encode categoricals)."""
        # Sort by Year
        if 'Year' in self.df.columns:
            self.df.sort_values('Year', inplace=True)

        feature_cols = [
            'War_Binary', 'War_Intensity', 'Cumulative_Conflict_10y',
            'GDP_Growth_lag1', 'GDP_Growth_lag2', 'Inflation_lag1',
            'Trade_Openness', 'Govt_Expenditure_GDP', 'Log_GDP_PC',
            'Inflation_Vol_5y', 'Growth_Vol_5y',
            'Food_Imports_Pct', 'Trade_Volatility', 'Global_Conflict_Intensity',
            'XR_Volatility', 'REER_Volatility', 'FDI_Inflows_GDP'
        ]

        # Check availability
        feature_cols = [c for c in feature_cols if c in self.df.columns]

        # Add Income Code
        if 'Income_Group' in self.df.columns:
            self.df['Income_Code'] = self.df['Income_Group'].astype('category').cat.codes
            feature_cols.append('Income_Code')

        target_col = 'GDP_Growth'

        # Drop rows with missing values
        ml_data = self.df[feature_cols + [target_col, 'Year', 'ISO3']].dropna()

        return ml_data, feature_cols, target_col

    def train_xgboost(self):
        """Trains XGBoost with Time Series Cross-Validation."""
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
        model.fit(X, y) # Final fit
        return model, feature_cols, np.mean(scores)

    def run_counterfactual_clustering(self):
        """
        1. Clusters countries into 'Regimes' based on Volatility and Growth.
        2. Trains a classifier to predict Regime.
        3. Simulates 'Counterfactual Movement' by reducing volatility.
        """
        logger.info("Running Counterfactual Clustering...")

        # Select Features for Clustering (Regime Definition)
        # We want to cluster on Outcomes + Drivers: Growth, Inflation, FX Vol, FDI, Trade Vol
        cluster_vars = ['GDP_Growth', 'Inflation', 'XR_Volatility', 'FDI_Inflows_GDP', 'Trade_Volatility']
        cluster_vars = [c for c in cluster_vars if c in self.df.columns]

        # Aggregate by ISO3 (Mean over period, e.g., last 10 years or full)
        # Using full period mean to define "Country Archetypes"
        df_agg = self.df.groupby('ISO3')[cluster_vars].mean().dropna()

        if df_agg.empty:
            logger.warning("Not enough data for clustering.")
            return {}

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_agg)

        # Cluster (k=3: e.g., Stable, Volatile, Stagnant)
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        df_agg['Cluster'] = clusters

        # Analyze Clusters to identify "Bad" (High Volatility) vs "Good" (Low Volatility)
        cluster_stats = df_agg.groupby('Cluster').mean()
        # Find cluster with highest XR_Volatility
        bad_cluster_id = cluster_stats['XR_Volatility'].idxmax()
        # Find cluster with lowest XR_Volatility
        good_cluster_id = cluster_stats['XR_Volatility'].idxmin()

        logger.info(f"Bad Cluster (High Vol): {bad_cluster_id}, Good Cluster (Low Vol): {good_cluster_id}")

        # Train Classifier: Predict Cluster from Features
        # Using Random Forest
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(df_agg[cluster_vars], df_agg['Cluster'])

        # Counterfactual: Take 'Bad' Cluster countries and give them 'Good' Cluster Mean Volatility
        bad_countries = df_agg[df_agg['Cluster'] == bad_cluster_id].copy()

        if bad_countries.empty:
            logger.warning("No countries in Bad Cluster.")
            return {}

        original_count = len(bad_countries)

        # Apply Treatment: Set XR_Volatility to Good Cluster Mean
        target_vol = cluster_stats.loc[good_cluster_id, 'XR_Volatility']
        bad_countries['XR_Volatility'] = target_vol

        # Predict new clusters
        new_clusters = clf.predict(bad_countries[cluster_vars])

        # Count Migrations (moved away from bad_cluster_id)
        migrated_count = np.sum(new_clusters != bad_cluster_id)
        pct_migrated = (migrated_count / original_count) * 100

        results = {
            'cluster_stats': cluster_stats,
            'bad_cluster_id': bad_cluster_id,
            'good_cluster_id': good_cluster_id,
            'original_count': original_count,
            'migrated_count': migrated_count,
            'pct_migrated': pct_migrated,
            'model': clf,
            'scaler': scaler,
            'kmeans': kmeans
        }

        logger.info(f"Counterfactual: {pct_migrated:.2f}% of countries migrated from Cluster {bad_cluster_id} after volatility reduction.")

        return results

    def cluster_recovery_trajectories(self):
        """
        Identifies clusters of post-conflict growth trajectories (t to t+4).
        """
        logger.info("Clustering Recovery Trajectories...")

        # 1. Identify Conflict End Points (Transition War -> Peace)
        # Shift War Status
        self.df.sort_values(['ISO3', 'Year'], inplace=True)
        self.df['War_Next'] = self.df.groupby('ISO3')['War_Binary'].shift(-1)

        # We find Year T where War=1 and War_Next=0.
        conflict_ends = self.df[(self.df['War_Binary'] == 1) & (self.df['War_Next'] == 0)].copy()

        trajectories = []
        labels = []

        for idx, row in conflict_ends.iterrows():
            iso = row['ISO3']
            end_year = row['Year']

            # We want growth for [end_year+1, ..., end_year+5]
            # Verify data availability
            c_data = self.df[self.df['ISO3'] == iso].set_index('Year')

            years_needed = range(int(end_year)+1, int(end_year)+6)
            if all(y in c_data.index for y in years_needed):
                 growth_path = c_data.loc[years_needed, 'GDP_Growth'].values
                 trajectories.append(growth_path)
                 labels.append(f"{iso}_{end_year}")

        if not trajectories:
            logger.warning("No complete recovery trajectories found (5 years post-war).")
            return None

        X_traj = np.array(trajectories)
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_traj)

        # Return dataframe with labels and cluster
        result = pd.DataFrame({
            'ID': labels,
            'Cluster': clusters,
            'Trajectory': list(X_traj)
        })

        return result, kmeans

    def cluster_trade_patterns(self):
        """
        Clusters countries by their trade adjustment patterns (Volatility & Dependency).
        """
        logger.info("Clustering Trade Patterns...")

        # Features for clustering: Mean Food Dependency, Trade Volatility
        # Collapse to country level
        cols = ['Food_Imports_Pct', 'Trade_Volatility', 'Food_Trade_Volatility']
        cols = [c for c in cols if c in self.df.columns]

        if not cols:
            return None

        trade_features = self.df.groupby('ISO3')[cols].mean().dropna()

        if trade_features.empty:
            return None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(trade_features)

        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        trade_features['Trade_Cluster'] = clusters

        return trade_features.reset_index(), kmeans
