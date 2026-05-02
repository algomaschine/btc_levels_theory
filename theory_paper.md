# Liquidity Potential Model (LPM) v2 — Walk-Forward Levels + Purged CV + Fractional Differentiation

This is an updated, falsifiable theory of why historical “levels” create both **attraction** and **repulsion**, and how to build a **predictive** system without look-ahead.

Key upgrades vs v1:
- **Purged walk-forward + embargo** to reduce label leakage from overlapping outcomes.
- **Walk-forward level discovery** (levels computed only from history visible at test time).
- **Fractional Differentiation** (De Prado) feature transform to reduce non-stationarity while preserving long memory.

---

## 1) Mechanism recap
We model historical levels as latent liquidity/attention “basins”. Let $p_t$ be price (not log price).

Let $\mathcal{L}_t = \{\ell_{1,t}, \dots, \ell_{N,t}\}$ be the set of levels visible at time $t$.
Each level has strength $w_{i,t} \ge 0$ and width $\sigma_{i,t} > 0$.

Define liquidity potential:

$$
U(p,t) = \sum_{i=1}^{N} w_{i,t}\,\exp\left(-\frac{(p-\ell_{i,t})^2}{2\sigma_{i,t}^2}\right)
$$

### Drift implication
Price drift is shaped by the gradient of the potential:

$$
\mu(p,t) = -\kappa\,\frac{\partial U}{\partial p}(p,t)
$$

This yields the observed push/pull around levels.

---

## 2) Walk-forward levels (no look-ahead)
Levels must be computed using only the history available at the time.

Operationally, in fold $k$:
- Build levels from data up to $T^{(k)}_{\text{train,end}}$.
- Apply those levels to label/feature-engineer the subsequent test segment.

This avoids the common “future levels” bias.

---

## 3) Purged walk-forward + embargo (De Prado-style)
If labels use a forward horizon $h$ (e.g., 60 minutes), then samples overlap in time.

For a test window $[T_0, T_1]$, purge training samples whose label end time $t_i + h$ overlaps the test interval.
Optionally apply an embargo period $\Delta$ after the test window.

This reduces leakage and over-optimistic scores.

---

## 4) Fractional Differentiation (preserve memory)
Instead of log prices, we use fractional differentiation to create a more stationary series while preserving long-range dependence.

We compute fractional differenced price:

$$
\tilde p_t = \sum_{k=0}^{K} w_k(d)\,p_{t-k}, \quad w_0=1,\; w_k(d) = -w_{k-1}(d)\,\frac{d-k+1}{k}
$$

- $d \in (0,1)$ is the fractional order.
- We truncate at $K$ where weights become small (fixed-width fractional differencing).

In code we attempt to use the `fracdiff` package if installed; otherwise we fall back to an internal implementation.

---

## 5) Falsifiable hypotheses
Same as v1 but evaluated with stricter protocol:

- **H1:** conditional outcome probabilities around levels improve out-of-sample.
- **H2:** repeated touches reduce rejection probability (depletion).
- **H3:** multi-timeframe confluence improves reaction odds.

Falsification: if out-of-sample scores and economic performance are indistinguishable from block-permuted baselines.
