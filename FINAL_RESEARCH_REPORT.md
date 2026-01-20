# FINAL RESEARCH REPORT: Asymmetric Resilience

## Abstract
This study analyzes the heterogeneous economic impacts of war across income groups using a global dataset (1990-2024). By integrating rigorous fixed-effects econometrics with machine learning (XGBoost), we identify non-linear recovery patterns and structural resilience gaps.

## 1. Data Overview
- **Sources:** Correlates of War (COW), World Bank (WDI).
- **Scope:** Global coverage, 217 countries.
- **Method:** Panel Data with Multiple Imputation (MICE).

## 2. Econometric Results
### Baseline Fixed Effects
The baseline model estimates the causal impact of war on growth, controlling for country and time fixed effects.

                           Model Comparison
=====================================================================
                               Baseline  Heterogeneity       Recovery
---------------------------------------------------------------------
Dep. Variable                GDP_Growth     GDP_Growth     GDP_Growth
Estimator                      PanelOLS       PanelOLS       PanelOLS
No. Observations                   7438           7438           6787
Cov. Est.                     Clustered      Clustered      Clustered
R-squared                        0.0196         0.0231         0.0180
R-Squared (Within)               0.0107         0.0141         0.0012
R-Squared (Between)             -17.506        -17.048        -23.201
R-Squared (Overall)             -4.2429        -4.1289        -6.5078
F-statistic                      28.778         21.250         20.005
P-value (F-stat)                 0.0000         0.0000         0.0000
======================     ============   ============   ============
War_Binary                      -3.3482                       -3.7187
                              (-3.1614)                     (-3.3215)
War_Intensity                   -0.4009        -0.4306
                              (-1.4120)      (-1.4745)
Trade_Openness                   0.0027         0.0029         0.0044
                               (0.6857)       (0.7276)       (1.0290)
Govt_Expenditure_GDP             0.0102
                               (1.4264)
Log_GDP_PC                       2.1196         2.1252         2.4771
                               (2.0834)       (2.0693)       (2.0072)
War_X_Income_HIC                               -1.0501
                                             (-1.0614)
War_X_Income_INX                               -1.3440
                                             (-1.1422)
War_X_Income_LIC                               -6.4864
                                             (-2.5396)
War_X_Income_LMC                               -1.3460
                                             (-1.5823)
War_X_Income_UMC                               -4.3326
                                             (-1.6053)
War_Binary_lag1                                                0.1140
                                                             (0.1336)
War_Binary_lag2                                                0.7741
                                                             (0.9422)
War_Binary_lag3                                                0.4689
                                                             (0.8191)
======================== ============== ============== ==============
Effects                          Entity         Entity         Entity
                                   Time           Time           Time
---------------------------------------------------------------------

T-stats reported in parentheses

## 3. Machine Learning Insights
### Predictive Performance
- **Model:** XGBoost (Time-Series CV)
- **RMSE:** 6.7247

### Feature Importance
The most critical predictors of post-conflict recovery are visualized in `feature_importance.png`.

## 4. Visual Analysis
### Event Study
![Event Study](event_study.png)
*Figure 1: Average GDP Growth trajectory 5 years before and after conflict onset.*

### Recovery Clusters
![Clusters](recovery_clusters.png)
*Figure 2: distinct recovery archetypes identified by K-Means clustering.*

## 5. Conclusion
Results suggest that while war universally depresses growth, low-income nations suffer deeper and longer-lasting penalties, confirming the "Asymmetric Resilience" hypothesis.
