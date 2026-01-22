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
    Runs Fixed Effects Panel Regressions for the 'War as Variance Amplifier' thesis.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # Ensure MultiIndex for PanelOLS
        if 'ISO3' in self.df.columns and 'Year' in self.df.columns:
            self.df = self.df.set_index(['ISO3', 'Year'])

    def run_horse_race_model(self):
        """
        The Core Tension Test: Institutions (Income Group) vs. Volatility (XR/Inflation).
        Model: Growth ~ War + War*LIC + War*XR_Vol + Controls
        """
        logger.info("Running Horse Race (Institutions vs Volatility)...")

        # Ensure interactions exist (created in data_processing, but safe check)
        # War_X_LIC, War_X_XR_Vol

        exog_vars = ['War_Binary', 'War_X_LIC', 'War_X_XR_Vol', 'Trade_Openness', 'Log_GDP_PC']
        # Add basic volatility if needed to control for level effects?
        # Usually: Y = Beta1*D + Beta2*D*M + Beta3*M. Interaction requires main effect of M.
        # Income_Group_Low is time-invariant, absorbed by FE.
        exog_vars += ['XR_Volatility']

        exog_vars = [v for v in exog_vars if v in self.df.columns]

        model_data = self.df.dropna(subset=['GDP_Growth'] + exog_vars)

        mod = PanelOLS(model_data['GDP_Growth'], model_data[exog_vars], entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=False)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        return res

    def run_hierarchy_test(self):
        """
        Stepwise regression to establish Mechanism Hierarchy.
        Returns a dictionary of results for comparison.
        """
        logger.info("Running Mechanism Hierarchy Test...")
        results = {}

        # Base Controls
        controls = ['Log_GDP_PC', 'Govt_Expenditure_GDP']
        controls = [c for c in controls if c in self.df.columns]

        # Step 0: Baseline (War Only)
        vars_0 = ['War_Binary'] + controls
        data_0 = self.df.dropna(subset=['GDP_Growth'] + vars_0)
        res_0 = PanelOLS(data_0['GDP_Growth'], data_0[vars_0], entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)
        results['Step0_Baseline'] = res_0

        # Step 1: + Upstream (FX Volatility)
        # We include main effect XR_Volatility. Interaction is for heterogeneity, here we test mediation/mechanism.
        # If War affects Growth VIA FX Volatility, adding FX Volatility should attenuate War coefficient.
        vars_1 = vars_0 + ['XR_Volatility']
        vars_1 = [v for v in vars_1 if v in self.df.columns]
        data_1 = self.df.dropna(subset=['GDP_Growth'] + vars_1)
        res_1 = PanelOLS(data_1['GDP_Growth'], data_1[vars_1], entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)
        results['Step1_Add_FX'] = res_1

        # Step 2: + Intermediate (Trade/Food)
        # Using High_Food_Import_Dep or Trade_Volatility
        # Let's use Trade_Volatility if available, or Food_Imports_Pct
        trade_var = 'Trade_Volatility' if 'Trade_Volatility' in self.df.columns else 'Food_Imports_Pct'
        vars_2 = vars_1 + [trade_var]
        vars_2 = [v for v in vars_2 if v in self.df.columns]
        data_2 = self.df.dropna(subset=['GDP_Growth'] + vars_2)
        res_2 = PanelOLS(data_2['GDP_Growth'], data_2[vars_2], entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)
        results['Step2_Add_Trade'] = res_2

        # Step 3: + Downstream (FDI)
        # FDI Inflows
        fdi_var = 'FDI_Inflows_GDP'
        vars_3 = vars_2 + [fdi_var]
        vars_3 = [v for v in vars_3 if v in self.df.columns]
        data_3 = self.df.dropna(subset=['GDP_Growth'] + vars_3)
        res_3 = PanelOLS(data_3['GDP_Growth'], data_3[vars_3], entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)
        results['Step3_Add_FDI'] = res_3

        return results

    def run_event_study(self):
        """
        Event Study with Leads and Lags.
        Lags: t-5 to t-1 (Leads in regression terms, anticipating war) -> We test Granger causality / parallel trends.
        Lags: t+1 to t+5 (Lags in regression terms, post war).
        Currently we only have 'War_Binary'. We need to construct event dummies.
        Since that's complex to do on the fly, we will use the existing Lags constructed in data_processing (lag1, lag2, lag3).
        And we will construct Leads here if possible or skip leads and focus on lag dynamics (persistence).
        """
        logger.info("Running Event Study (Lags only)...")

        # Use available lags
        lags = [c for c in self.df.columns if 'War_Binary_lag' in c] # lag1, lag2, lag3
        # Baseline War_Binary (t=0)

        exog_vars = ['War_Binary'] + lags + ['Cumulative_Conflict_10y', 'Log_GDP_PC', 'Trade_Openness']
        exog_vars = [v for v in exog_vars if v in self.df.columns]

        model_data = self.df.dropna(subset=['GDP_Growth'] + exog_vars)

        mod = PanelOLS(model_data['GDP_Growth'], model_data[exog_vars], entity_effects=True, time_effects=True, drop_absorbed=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)

        return res

    def run_heterogeneity_models(self):
        """
        Runs the two heterogeneity dimensions: Food and Income.
        """
        logger.info("Running Heterogeneity Models...")
        results = {}

        # Model A: Food Import Dependence
        # Interaction: War * High_Food_Import_Dep
        # Main effects: War, High_Food_Import_Dep
        vars_a = ['War_Binary', 'War_X_HighFoodImp', 'High_Food_Import_Dep', 'Log_GDP_PC']
        vars_a = [v for v in vars_a if v in self.df.columns]
        data_a = self.df.dropna(subset=['GDP_Growth'] + vars_a)
        res_a = PanelOLS(data_a['GDP_Growth'], data_a[vars_a], entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=False).fit(cov_type='clustered', cluster_entity=True)
        results['Heterogeneity_Food'] = res_a

        # Model B: Income Group
        # Interaction: War * Low_Income
        # Main effects: War (Income_Group_Low is absorbed by FE)
        vars_b = ['War_Binary', 'War_X_LIC', 'Log_GDP_PC']
        vars_b = [v for v in vars_b if v in self.df.columns]
        data_b = self.df.dropna(subset=['GDP_Growth'] + vars_b)
        res_b = PanelOLS(data_b['GDP_Growth'], data_b[vars_b], entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=False).fit(cov_type='clustered', cluster_entity=True)
        results['Heterogeneity_Income'] = res_b

        return results

    def save_results(self, all_results):
        """
        Saves all results to txt and tex.
        all_results: dict of result objects.
        """
        logger.info("Saving Econometric Results...")

        with open("ECONOMETRIC_RESULTS.txt", "w") as f:
            for name, res in all_results.items():
                f.write(f"--- {name} ---\n")
                f.write(str(res))
                f.write("\n\n")

        # Create comparison table for Main Results
        # Filter keys for the main table (Hierarchy steps or Horse Race)
        main_keys = [k for k in all_results.keys() if 'Step' in k or 'Horse' in k]
        main_dict = {k: all_results[k] for k in main_keys}

        if main_dict:
            comparison = compare(main_dict)
            with open("REGRESSION_TABLE.tex", "w") as f:
                f.write(comparison.summary.as_latex())

        return
