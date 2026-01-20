import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from linearmodels.panel import compare
import logging
import tabulate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EconometricModeler:
    """
    Runs Fixed Effects Panel Regressions and Hypothesis Tests.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # Set index for PanelOLS
        if 'ISO3' in self.df.columns and 'Year' in self.df.columns:
            self.df = self.df.set_index(['ISO3', 'Year'])

    def run_baseline_model(self):
        """
        Model 1: Baseline Impact of War on Growth
        Equation: Growth_it = beta * War_Binary_it + Controls + alpha_i + delta_t + epsilon_it
        """
        logger.info("Running Baseline Model...")

        exog_vars = ['War_Binary', 'War_Intensity', 'Trade_Openness', 'Govt_Expenditure_GDP', 'Log_GDP_PC']
        # Add constant
        exog_vars = [v for v in exog_vars if v in self.df.columns]

        # Check for missing values in exog or endog
        model_data = self.df.dropna(subset=['GDP_Growth'] + exog_vars)

        # PanelOLS with Entity and Time Effects
        mod = PanelOLS(model_data['GDP_Growth'], model_data[exog_vars], entity_effects=True, time_effects=True, drop_absorbed=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)

        return res

    def run_heterogeneity_model(self):
        """
        Model 2: Heterogeneity by Income Group
        Interaction: War_Binary * Income_Group (Categorical)
        """
        logger.info("Running Heterogeneity Model...")

        # Create Dummy Interactions manually if needed, or use formula if supported (PanelOLS formula is distinct)
        # We'll create dummies for Income Groups and interact.

        # We need to drop one base group (e.g., High Income)
        # But Income_Group is a string column.

        # Create dummies
        # Reset index to access columns easily
        df_reset = self.df.reset_index()

        # One-hot encode Income Group
        dummies = pd.get_dummies(df_reset['Income_Group'], prefix='Income')
        df_reset = pd.concat([df_reset, dummies], axis=1)

        # Interaction terms
        interaction_cols = []
        for col in dummies.columns:
            inter_col = f'War_X_{col}'
            df_reset[inter_col] = df_reset['War_Binary'] * df_reset[col]
            interaction_cols.append(inter_col)

        # Set index back
        model_df = df_reset.set_index(['ISO3', 'Year'])

        base_vars = ['War_Intensity', 'Trade_Openness', 'Log_GDP_PC']
        # Include interactions, drop base dummy interaction to avoid collinearity if War_Binary is included?
        # Actually, if we include War_Binary, we should drop one interaction.
        # Let's include all interactions and drop War_Binary (or vice versa).
        # Common approach: War_Binary + War_Binary * Low_Income + War_Binary * Upper_Middle... (High Income is base)

        # Identify groups
        # Groups: 'High income', 'Low income', 'Lower middle income', 'Upper middle income'
        # Base: High income.

        # Use all interactions to show effect per group, remove War_Binary to avoid perfect collinearity
        # Growth ~ War_High + War_Low + ... + Controls

        exog_vars = interaction_cols + base_vars
        exog_vars = [v for v in exog_vars if v in model_df.columns]

        model_data = model_df.dropna(subset=['GDP_Growth'] + exog_vars)

        # Check rank=False to proceed if some groups have no wars (all zeros)
        mod = PanelOLS(model_data['GDP_Growth'], model_data[exog_vars], entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=False)
        res = mod.fit(cov_type='clustered', cluster_entity=True)

        return res

    def run_recovery_model(self):
        """
        Model 3: Recovery Dynamics (Lags)
        """
        logger.info("Running Recovery Model...")

        lags = [c for c in self.df.columns if 'War_Binary_lag' in c]
        controls = ['Trade_Openness', 'Log_GDP_PC']
        exog_vars = ['War_Binary'] + lags + controls
        exog_vars = [v for v in exog_vars if v in self.df.columns]

        model_data = self.df.dropna(subset=['GDP_Growth'] + exog_vars)

        mod = PanelOLS(model_data['GDP_Growth'], model_data[exog_vars], entity_effects=True, time_effects=True, drop_absorbed=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)

        return res

    def save_results(self, results_dict):
        """
        Saves regression results to a text file and LaTeX.
        """
        logger.info("Saving Econometric Results...")

        with open("ECONOMETRIC_RESULTS.txt", "w") as f:
            for name, res in results_dict.items():
                f.write(f"--- {name} ---\n")
                f.write(str(res))
                f.write("\n\n")

        # Compare models table
        # linearmodels.panel.compare
        comparison = compare(results_dict)

        with open("REGRESSION_TABLE.tex", "w") as f:
            # simple to_latex equivalent if available or manual
            # linearmodels summary has as_latex()? No, straightforward access is usually via summary.
            # compare object has summary.
            f.write(comparison.summary.as_latex())

        return comparison

if __name__ == "__main__":
    pass
