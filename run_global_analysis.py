import logging
import sys
import os
import pandas as pd
from src.data_ingestion import ConflictDownloader, WorldBankFetcher
from src.data_processing import DataProcessor
from src.econometrics import EconometricModeler
from src.ml_models import ConflictML
from src.reporting import Reporter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.StreamHandler(sys.stdout),
    logging.FileHandler("analysis.log")
])
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Global Analysis Pipeline...")

    # 1. Data Ingestion
    logger.info("--- Step 1: Data Ingestion ---")
    try:
        # Conflict Data
        cd = ConflictDownloader()
        conflict_df = cd.process_cow_data()
        logger.info(f"Conflict Data Downloaded: {conflict_df.shape}")

        # World Bank Data
        wb = WorldBankFetcher()
        meta_df = wb.fetch_metadata()
        econ_df = wb.fetch_indicators(start_year=1990, end_year=2024)
        logger.info(f"Economic Data Downloaded: {econ_df.shape}")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return

    # 2. Data Processing
    logger.info("--- Step 2: Data Processing ---")
    dp = DataProcessor(conflict_df, econ_df, meta_df)
    dp.merge_data()
    dp.impute_missing_values()
    dp.engineer_features()
    clean_df = dp.get_clean_panel()

    # Save processed data
    clean_df.to_csv("processed_global_data.csv", index=False)
    logger.info(f"Processed Data Saved: {clean_df.shape}")

    # 3. Econometric Analysis
    logger.info("--- Step 3: Econometric Analysis ---")
    em = EconometricModeler(clean_df)

    res_baseline = em.run_baseline_model()
    res_hetero = em.run_heterogeneity_model()
    res_recovery = em.run_recovery_model()

    results_dict = {
        'Baseline': res_baseline,
        'Heterogeneity': res_hetero,
        'Recovery': res_recovery
    }

    comparison = em.save_results(results_dict)

    # 4. Machine Learning
    logger.info("--- Step 4: Machine Learning ---")
    ml = ConflictML(clean_df)

    # XGBoost
    xgb_model, feature_names, rmse = ml.train_xgboost()

    # Clustering
    clusters_df, kmeans_model = ml.cluster_recovery_trajectories()

    # 5. Reporting
    logger.info("--- Step 5: Reporting ---")
    reporter = Reporter(clean_df)

    reporter.plot_event_study()
    reporter.plot_feature_importance(xgb_model, feature_names)
    reporter.plot_recovery_clusters(clusters_df)

    reporter.generate_report_md(str(comparison), ml_rmse=rmse)
    reporter.generate_latex()

    logger.info("Pipeline Completed Successfully.")

if __name__ == "__main__":
    main()
