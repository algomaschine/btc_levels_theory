# A Falsifiable, Predictive Theory of “Support/Resistance” as Market Memory (Liquidity Potential Model)

**Scope**: This paper proposes a **predictive and falsifiable** mechanism explaining why prices repeatedly **approach** ("attract") and sometimes **reverse** ("repel") around historical price zones commonly called *support* and *resistance*.

It also addresses your trauma metaphor and provides a practical research/backtest framework for **BTCUSDT 1‑minute OHLCV** data.

> Important: This is a scientific-style model. It does **not** assume a mystical “collective unconscious.” It treats “memory” as an emergent consequence of (i) **order clustering**, (ii) **position/inventory overhang**, (iii) **behavioral anchoring**, and (iv) **risk constraints / liquidation mechanics**.

---

## 1. The core phenomenon: attraction + repulsion near historical levels

Market practitioners observe:

- **Attraction**: price tends to revisit prior swing highs/lows, volume nodes, round numbers.
- **Repulsion**: price frequently reverses at those zones, at least temporarily.
- **Regime dependence**: in trends, “resistance becomes support” (role reversal) and levels break; in ranges they hold.

### Your point (1): contraction in reactions

A typical empirical pattern is **diminishing reaction magnitude** after repeated tests:

- First touch: large bounce/rejection.
- Later touches: smaller bounce.
- Eventually: break (or the level becomes irrelevant).

This can be modeled as **liquidity depletion** and/or **information assimilation**.

---

## 2. Is the trauma metaphor correct? (Your point 2)

As a metaphor it is **useful but risky**.

**Useful** because both systems show:
- **Approach–avoidance conflict**: moving toward a “charged” region while also resisting it.
- **Habituation / desensitization**: repeated exposure can reduce reaction amplitude.
- **Context dependence**: reactions differ depending on the broader regime (stress, trend, volatility).

**Risky** because:
- In humans, “trauma” involves **meaning, narrative, physiology**.
- In markets, the mechanism is explainable via **microstructure and incentives** without invoking unconscious archetypes.

So: metaphor is fine **as an intuition**. For trading you want a mechanism grounded in measurable variables.

---

## 3. A shared mechanism with a name (Your point 3)

### 3.1 Behavioral mechanism (humans)
A known construct is **approach–avoidance conflict** (Lewin), plus:
- **conditioning** and **extinction / exposure** (habituation)
- **prediction error minimization** (Bayesian brain / active inference)

### 3.2 Market mechanism (microstructure + behavior)
The market analog is **memory via latent liquidity + anchoring**:

1) **Anchoring**: traders reference salient prior prices (“I bought at 40k”).
2) **Inventory overhang**: trapped longs/shorts want to exit at breakeven.
3) **Stop clustering**: protective stops and breakout stops accumulate around known levels.
4) **Liquidation / margin dynamics**: leverage creates “forced flows” near certain thresholds.
5) **Market making / adverse selection**: liquidity provision is sensitive to known boundaries.

This is often described as:
- **liquidity pools** around levels
- **order clustering**
- **support/resistance as supply/demand zones**
- **role reversal** as the order book resets after a break

---

## 4. Proposed predictive, falsifiable theory: Liquidity Potential Model (LPM)

### 4.1 Definitions
Let price at time *t* be \(p_t\) (use log-price in practice).

Let \(\mathcal{L}_t = \{\ell_{1,t}, \dots, \ell_{N,t}\}\) be the set of *active levels* derived from higher timeframes (pivots + volume profile peaks).

Each level has a **strength** \(w_{i,t} \ge 0\) and a **width** \(\sigma_{i,t} > 0\).

Define a *liquidity potential*:

\[
U(p,t) = \sum_{i=1}^{N} w_{i,t}\,\exp\left(-\frac{(p-\ell_{i,t})^2}{2\sigma_{i,t}^2}\right)
\]

Interpretation:
- Where there is more latent liquidity/attention, \(U\) is higher.
- Its gradient produces a **mean drift** toward/away depending on position.

### 4.2 Price dynamics (SDE)
Model short-horizon returns as a diffusion with drift from the potential:

\[
\mathrm{d}p_t = -\kappa\,\frac{\partial U}{\partial p}(p_t,t)\,\mathrm{d}t + \eta_t\,\mathrm{d}t + \sigma_t\,\mathrm{d}W_t
\]

Where:
- \(\kappa > 0\) sets how strongly levels affect drift.
- \(\eta_t\) captures slow trend components (estimated from higher timeframe).
- \(\sigma_t\) is local volatility.

Compute gradient:

\[
\frac{\partial U}{\partial p}(p,t) = \sum_i w_{i,t}\,\exp\left(-\frac{(p-\ell_i)^2}{2\sigma_i^2}\right)\,\left(-\frac{(p-\ell_i)}{\sigma_i^2}\right)
\]

Thus drift term is:

\[
\mu(p,t) = -\kappa\,\frac{\partial U}{\partial p}(p,t)
= \kappa\sum_i w_{i,t}\,\exp\left(-\frac{(p-\ell_i)^2}{2\sigma_i^2}\right)\,\frac{(p-\ell_i)}{\sigma_i^2}
\]

**Key implication**: drift changes sign across the level—creating the observed “push/pull.”

### 4.3 Dynamics of level strength (habituation / depletion)
To capture “contraction” you hypothesize that level strength evolves with touches:

\[
\Delta w_{i,t} = +\alpha\,\text{(attention / volume formation)} - \beta\,\text{(touch depletion)} - \gamma\,\text{(time decay)}
\]

Operational version (discrete):

- If a level is formed by high volume at price: increase \(w\).
- Each time price **touches** the level (within tolerance), decrease \(w\) (liquidity consumed).
- Over time, decay \(w\) unless reinforced.

This encodes:
- repeated tests -> weaker reaction
- long time without touch -> level relevance decays

### 4.4 Falsifiable hypotheses

H1 (**predictive drift**): Conditional expected forward return has the sign predicted by \(\mu(p,t)\) after controlling for volatility/trend.

H2 (**depletion**): After *k* touches, probability of rejection decreases and probability of breakout increases (all else equal).

H3 (**multi-timeframe confluence**): Levels supported by multiple higher timeframes have higher rejection probability and/or larger expected move.

H4 (**volume-node levels**): Levels near volume profile peaks have higher reaction probability than random price points.

**Falsification**: if in out-of-sample tests these effects are not statistically above a null bootstrap/permutation baseline, LPM is rejected.

---

## 5. Murray Math levels (Your point 4)

“Murray Math Lines” are popular in retail technical analysis and loosely inspired by Gann-style octave partitioning.

- They are **not** broadly supported by peer‑reviewed evidence as a universal market law.
- They can still be tested as a feature set: “Does discretizing price into octave levels yield predictive power vs. baselines?”

In LPM terms, Murray levels are a **hand-crafted prior** on \(\ell_i\). LPM lets you test:
- whether those \(\ell_i\) outperform data-driven level discovery (pivots + volume nodes)
- whether they add incremental predictive power in a model

---

## 6. Practical trading implications (what can be predictive)

Under LPM, you do not treat a level as “magic.” You trade **conditional probabilities**:

- **Rejection setup**: if price enters a level band and LPM predicts repulsion (drift away) with high probability.
- **Breakout setup**: if repeated touches + trend + high volatility indicates depletion and higher breakout odds.

Prediction target examples:
- \(P(\text{reversal} | \text{touch}, \text{features})\)
- \(P(\text{breakout} | \text{touch}, \text{features})\)
- \(E[r_{t\to t+h}]\) conditional on distance to confluence levels

---

## 7. Empirical protocol on BTCUSDT 1‑minute data

We will:
1) Build **higher timeframe bars** (5m/15m/1h/4h/1d).
2) Detect levels from each timeframe:
   - pivot highs/lows (fractal)
   - volume profile peaks (price–volume histogram)
3) Cluster levels into zones with tolerance proportional to volatility.
4) For each minute, compute features relative to the nearest levels.
5) Label outcomes (rejection vs breakout) using forward returns and crossing rules.
6) Fit a **walk-forward** predictive model (logistic regression baseline).
7) Backtest a simple strategy using the predicted probabilities.

---

## 8. Deliverables in this repo

- `src/` implements the full pipeline
- `run.py` runs:
  - level discovery
  - hypothesis tests
  - walk-forward modeling
  - strategy backtest
  - output report under `./output/`

---

## 9. Limitations

- With OHLCV only, you do not observe the full order book.
- Results can still be meaningful because BTC has strong behaviorally-driven clustering.
- Beware overfitting: use purged walk-forward and strict out-of-sample windows.

