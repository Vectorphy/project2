# Sectioned Analytical Notes: War as a Variance Amplifier

## 1. Research Tension
**The Debate:**
The economic literature on conflict recovery is divided between two dominant views:
1.  **The Institutional Scarring View (Endowment):** War destroys the "rules of the game" and physical capital. Recovery is a function of initial institutional quality and endowment.
    *   *Prediction:* Recovery is slow and path-dependent. High-Income Countries (HICs) with strong institutions recover; Low-Income Countries (LICs) suffer permanent level shifts. The primary driver is *who you are*.
2.  **The Variance Amplification View (Shock):** War is primarily a massive volatility shock (uncertainty). It spikes risk premia (FX, Inflation), which paralyzes downstream activity (Trade, Investment).
    *   *Prediction:* The growth penalty is driven by *exposure* to volatility. If volatility is tamed, growth recovers, regardless of deep institutional quality. The primary driver is *what you face*.

**Core Research Question:**
*"Is the post-war growth penalty primarily a structural deficit determined by institutional endowments, or a liquidity constraint driven by macroeconomic volatility?"*

**The Test (The "Horse Race"):**
Run a Fixed Effects model competing these interactions:
$$ Growth_{it} = \beta_1 War_{it} + \beta_2 (War_{it} \times IncomeGroup_{i}) + \beta_3 (War_{it} \times VolatilityMetric_{it}) + \dots $$
*   If $\beta_2$ dominates: Institutional view holds.
*   If $\beta_3$ dominates: Variance view holds.

---

## 2. Mechanism Hierarchy
To move beyond a "laundry list" of variables, we impose a causal ordering based on the speed of transmission.

**Proposed Hierarchy (Upstream $\to$ Downstream):**
1.  **Level 1 (Immediate/Upstream):** **Financial Volatility** (Exchange Rate Volatility, Inflation Volatility). These respond instantly to conflict onset due to capital flight and uncertainty.
2.  **Level 2 (Intermediate):** **Trade Disruption** (Food Import dependence, Trade Openness). Volatility raises transaction costs, breaking trade links.
3.  **Level 3 (Downstream):** **Investment Collapse** (FDI Inflows). Investment is irreversible and sensitive to the uncertainty generated in Levels 1 & 2.
4.  **Outcome:** **GDP Growth**.

**The Ranking Test:**
A stepwise "attribution" test. We observe the attenuation of the War coefficient as we add mechanisms in order.
*   *Baseline:* $Growth \sim War$ (Total Effect).
*   *Step 1:* $Growth \sim War + FX\_Vol$. (How much does FX explain?)
*   *Step 2:* $Growth \sim War + FX\_Vol + Trade$. (Incremental explainability of Trade).
*   *Step 3:* $Growth \sim War + FX\_Vol + Trade + FDI$. (Incremental explainability of Investment).

*Hypothesis:* The largest drop in the War coefficient will occur at Step 1 (FX/Variance), identifying it as the "Master Mechanism."

---

## 3. Time Dynamics Theory
**Economic Justification:**
*   **Short-Run (Shock, $t+0$ to $t+2$):** Dominated by the **Variance Channel**. Uncertainty spikes, capital flees, liquidity dries up. Theory predicts sharp negative growth but potential for rapid "bounce back" if variance subsides.
*   **Medium-Run (Adjustment, $t+3$ to $t+5$):** Dominated by the **Trade/Investment Channel**. Contracts are renegotiated, supply chains re-route.
*   **Long-Run (Scarring, $t>5$):** Dominated by **Institutional Channel**. Capital destruction (human and physical) lowers potential output.

**Event Study Structure:**
$$ Growth_{it} = \alpha_i + \gamma_t + \sum_{k=-5}^{5} \delta_k D_{it}^k + X_{it}\beta + \epsilon_{it} $$
Where $D_{it}^k$ are leads and lags of conflict onset.

**Implications of Coefficients:**
*   $\delta_0 \ll 0$: The immediate shock.
*   $\delta_{1,2,3} < 0$: Persistence of the variance shock.
*   $\delta_{4,5} \approx 0$: Recovery (V-shaped).
*   $\delta_{4,5} < 0$: Permanent scarring (L-shaped).

---

## 4. Heterogeneity with Theory
We examine two dimensions to test the competing theories.

**Dimension 1: Food Import Dependence (The Variance Exposure)**
*   *Theory:* High dependence on essential imports (food) implies low elasticity of substitution. Currency volatility translates directly into domestic inflation and consumption shocks.
*   *Hypothesis:* The conflict penalty is non-linearly higher for countries with Food Imports > 20% of merchandise imports.
*   *Test:* Interaction $War \times HighFoodImp$ is negative and significant.

**Dimension 2: Income Group (The Institutional Endowment)**
*   *Theory:* HICs have "fiscal space" and institutional credibility to smooth shocks (counter-cyclical policy). LICs are pro-cyclical.
*   *Hypothesis:* $War \times LIC$ shows a deeper penalty than $War \times HIC$.

**Synthesis:**
If Food Import Dependence remains significant even when controlling for Income Group, it proves that *structural vulnerability to variance* matters independently of general institutional quality.

---

## 5. Clustering Justification
**Why Clustering?**
Regressions estimate *average marginal effects* (holding all else constant). But conflict recovery is a "syndrome" – a bundle of co-moving variables (High Volatility + Low FDI + High Inflation). Clustering identifies these **multidimensional archetypes** (e.g., "The Variance Trap" vs. "The Resilient Rebounder").

**Counterfactual Cluster Movement:**
This analysis answers: *"What if?"*
*   *Definition:* If we take a country in the "Conflict Trap" cluster and artificially set its **FX Volatility** to the median of the "Peaceful" cluster (holding other variables constant), does it mathematically migrate to a better recovery cluster?
*   *Meaning:* This quantifies the specific contribution of a mechanism to the *overall regime* of the country.

**Recommendation:**
Place the **Cluster Archetypes** (Centroids) in the Main Text to illustrate the "types" of recovery. Place the detailed silhouette scores and method in the Appendix.

---

## 6. Contribution Statement
The conflict literature typically assumes war is a *level shock* to capital stocks, where recovery is a slow convergence process dictated by institutional quality. **This paper shows instead** that war is primarily a **variance amplifier**, where the transmission of shock occurs through rapid-response channels like currency volatility and trade disruption. **This changes how we think about recovery** because it implies that stabilizing nominal volatility (FX, Inflation) is a prerequisite for real recovery, potentially more urgent than deep structural institutional reform in the immediate post-war period.

---

## 7. Policy Discipline

| Policy Implication | Empirical Support | Targeted Mechanism | Failure Condition |
| :--- | :--- | :--- | :--- |
| **Currency Peg / Dollarization** | Strong "Currency Channel" ranking; High coefficients on FX Volatility. | **FX Volatility (Upstream)** | **Loss of Autonomy:** If the shock requires real depreciation to restore competitiveness, a hard peg prevents adjustment and causes stagnation. |
| **Trade Guarantee Funds** | Significant interaction of $War \times FoodImportDep$. | **Trade Disruption (Intermediate)** | **Physical Destruction:** Guarantees cannot move goods if ports/roads are physically destroyed (supply side constraints). |
| **FDI Risk Insurance (MIGA)** | FDI identified as a downstream victim of Upstream Volatility. | **Investment Collapse (Downstream)** | **Institutional Threshold:** Insurance covers political risk, but not operational unviability due to lack of local skilled labor or power (structural gaps). |
