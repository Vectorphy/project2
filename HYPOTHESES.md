# Research Hypotheses
## Hypothesis 1: The "Conflict Penalty"
**Statement:** Active conflict significantly reduces GDP growth, even after controlling for time-invariant country characteristics and global shocks.
**Test:** Coefficient of `War_Binary` in the Baseline Fixed Effects Model is negative and statistically significant ($p < 0.05$).

## Hypothesis 2: Asymmetric Resilience
**Statement:** Low-income countries suffer a larger growth penalty from conflict compared to High-income countries, due to lower fiscal and institutional resilience.
**Test:** The interaction term `War_Binary * Low Income` is negative and significant, indicating a larger penalty than the base group (High Income).

## Hypothesis 3: The "Recovery Lag"
**Statement:** The negative economic effects of conflict persist for at least 3 years after the conflict onset.
**Test:** Coefficients of `War_Binary_lag1`, `War_Binary_lag2`, and `War_Binary_lag3` are negative and significant in the Dynamic Model.

## Hypothesis 4: Intensity Matters
**Statement:** The economic damage is proportional to conflict intensity (battle deaths).
**Test:** Coefficient of `War_Intensity` (Log Deaths) is negative and significant.

## Hypothesis 5: The "Trade Amplifier"
**Statement:** High dependency on food imports amplifies the negative impact of war on growth.
**Test:** The coefficient of the interaction term `War_Binary * Food_Imports_Pct` is negative and significant.

## Hypothesis 6: Global Spillovers
**Statement:** Global conflict intensity negatively impacts the growth of peaceful nations, specifically those with high food import dependency.
**Test:** In the non-war sample, the coefficient of `Global_Conflict_Intensity * Food_Imports_Pct` is negative and significant.
