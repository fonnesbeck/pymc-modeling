# Diagnostics and Visualization

## Table of Contents
- [Quick Diagnostic Checklist](#quick-diagnostic-checklist)
- [Convergence Diagnostics](#convergence-diagnostics)
- [ArviZ Plotting Reference](#arviz-plotting-reference)
- [Divergence Troubleshooting](#divergence-troubleshooting)
- [Model Comparison](#model-comparison)

## Quick Diagnostic Checklist

After every sampling run:

```python
import arviz as az

# 1. Check for divergences
n_div = idata.sample_stats["diverging"].sum().item()
print(f"Divergences: {n_div}")

# 2. Summary statistics with convergence diagnostics
summary = az.summary(idata, var_names=["~offset"])  # exclude auxiliary params
print(summary[["mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "ess_tail", "r_hat"]])

# 3. Quick visual check
az.plot_trace(idata, compact=True)
```

**Pass criteria**:
- Zero divergences (or < 0.1% and investigated)
- `r_hat < 1.01` for all parameters
- `ess_bulk > 400` and `ess_tail > 400` (total across chains)
- Trace plots show good mixing (caterpillar appearance)

## Convergence Diagnostics

### R-hat (Potential Scale Reduction Factor)

Compares between-chain and within-chain variance.

```python
az.rhat(idata)

# Flag problematic parameters
rhat_df = az.summary(idata)
problematic = rhat_df[rhat_df["r_hat"] > 1.01]
```

**Interpretation**:
- `r_hat = 1.0`: Perfect convergence
- `r_hat < 1.01`: Good (strict threshold)
- `r_hat < 1.05`: Acceptable (permissive)
- `r_hat > 1.1`: Not converged

### Effective Sample Size (ESS)

Accounts for autocorrelation in MCMC samples.

```python
az.ess(idata, method="bulk")  # center of distribution
az.ess(idata, method="tail")  # extremes, important for credible intervals
```

**Interpretation**:
- `ess > 400`: adequate for most purposes
- `ess > 1000`: publication-quality
- `ess < 100`: unreliable

### Monte Carlo Standard Error (MCSE)

```python
az.mcse(idata)
```

MCSE should be < 5% of posterior SD.

---

## ArviZ Plotting Reference

### Posterior Visualization

#### plot_posterior
**Use case**: Summarize marginal posteriors with point estimates and credible intervals.

```python
# Basic posterior summary
az.plot_posterior(idata, var_names=["beta", "sigma"])

# With reference value (e.g., null hypothesis)
az.plot_posterior(idata, var_names=["beta"], ref_val=0)

# Show specific credible interval
az.plot_posterior(idata, hdi_prob=0.9)

# Combine with rope (region of practical equivalence)
az.plot_posterior(idata, var_names=["beta"], rope=[-0.1, 0.1])
```

#### plot_forest
**Use case**: Compare parameters across groups or models; ideal for hierarchical models.

```python
# Compare group-level parameters
az.plot_forest(idata, var_names=["alpha"], combined=True)

# Multiple models side by side
az.plot_forest([idata_1, idata_2], model_names=["Model 1", "Model 2"])

# With R-hat coloring to spot convergence issues
az.plot_forest(idata, var_names=["alpha"], r_hat=True)

# ESS displayed
az.plot_forest(idata, var_names=["alpha"], ess=True)
```

#### plot_ridge
**Use case**: Visualize distributions for many parameters compactly.

```python
# Ridge plot for group effects
az.plot_ridge(idata, var_names=["alpha"])

# Combined chains
az.plot_ridge(idata, var_names=["beta"], combined=True)
```

#### plot_violin
**Use case**: Compare distributions with quartile information.

```python
az.plot_violin(idata, var_names=["alpha"])
```

### Convergence Diagnostics Plots

#### plot_trace
**Use case**: Visual check for mixing and stationarity. First plot after sampling.

```python
# Basic trace plot (density + trace)
az.plot_trace(idata, var_names=["beta", "sigma"])

# Compact for many parameters
az.plot_trace(idata, compact=True, combined=True)

# Rank-normalized (more sensitive to convergence issues)
az.plot_trace(idata, kind="rank_bars")
az.plot_trace(idata, kind="rank_vlines")
```

**Look for**:
- Fuzzy caterpillar (good mixing)
- No trends (stationarity)
- Overlapping densities across chains

#### plot_rank
**Use case**: More sensitive than trace plots for detecting non-convergence.

```python
az.plot_rank(idata, var_names=["beta"])
```

**Look for**: Uniform distribution of ranks. Deviations indicate convergence problems.

#### plot_ess
**Use case**: Visualize how ESS accumulates; identify if more samples needed.

```python
# ESS evolution over iterations
az.plot_ess(idata, var_names=["beta"], kind="evolution")

# Local ESS (identifies problematic regions)
az.plot_ess(idata, var_names=["beta"], kind="local")

# Quantile ESS
az.plot_ess(idata, var_names=["beta"], kind="quantile")
```

#### plot_mcse
**Use case**: Visualize Monte Carlo error across the distribution.

```python
# MCSE for different quantiles
az.plot_mcse(idata, var_names=["beta"])

# Local MCSE
az.plot_mcse(idata, var_names=["beta"], kind="local")
```

#### plot_autocorr
**Use case**: Identify slow mixing; high autocorrelation means low ESS.

```python
az.plot_autocorr(idata, var_names=["beta"])
```

**Look for**: Rapid decay to zero. Slow decay indicates poor mixing.

#### plot_energy
**Use case**: Diagnose HMC/NUTS-specific issues.

```python
az.plot_energy(idata)
```

**Look for**: Overlapping marginal and transition energy distributions. Large gaps indicate poor exploration.

### Parameter Relationships

#### plot_pair
**Use case**: Identify correlations, multimodality, and divergence patterns.

```python
# Basic pair plot
az.plot_pair(idata, var_names=["alpha", "beta", "sigma"])

# With divergences highlighted (critical for debugging)
az.plot_pair(idata, var_names=["alpha", "sigma"], divergences=True)

# KDE instead of scatter
az.plot_pair(idata, var_names=["alpha", "beta"], kind="kde")

# Hexbin for large samples
az.plot_pair(idata, var_names=["alpha", "beta"], kind="hexbin")

# With marginals
az.plot_pair(idata, var_names=["alpha", "beta"], marginals=True)

# Reference point (e.g., true values in simulation)
az.plot_pair(idata, var_names=["alpha", "beta"], reference_values={"alpha": 1.0, "beta": 0.5})
```

#### plot_parallel
**Use case**: Visualize high-dimensional parameter space; spot divergences.

```python
# Highlight divergent samples
az.plot_parallel(idata, var_names=["alpha", "beta", "sigma"])
```

Divergent samples shown in different color—look for patterns indicating problematic regions.

### Model Checking

#### plot_ppc (Posterior Predictive Check)
**Use case**: Does the model capture the data-generating process?

```python
# First, generate posterior predictive samples
with model:
    idata.extend(pm.sample_posterior_predictive(idata))

# Density overlay (default)
az.plot_ppc(idata)

# Cumulative distribution (better for detecting systematic misfit)
az.plot_ppc(idata, kind="cumulative")

# Scatter plot (observed vs predicted)
az.plot_ppc(idata, kind="scatter")

# For specific observed variable
az.plot_ppc(idata, var_names=["y"])

# Subsample for speed with large datasets
az.plot_ppc(idata, num_pp_samples=100)
```

**Look for**: Observed data (dark line) within posterior predictive distribution (light lines).

#### plot_ppc with groups
**Use case**: Check model fit across subgroups.

```python
# Grouped PPC (requires coords)
az.plot_ppc(idata, kind="cumulative", flatten=[])  # separate by observation
```

#### plot_loo_pit
**Use case**: Calibration check using LOO probability integral transform.

```python
az.plot_loo_pit(idata, y="y")
```

**Look for**: Uniform distribution. U-shape indicates underdispersion; inverse-U indicates overdispersion.

#### plot_bpv (Bayesian p-values)
**Use case**: Quantile-based model check.

```python
az.plot_bpv(idata, kind="p_value")
az.plot_bpv(idata, kind="t_stat")  # test statistic version
```

**Look for**: Values near 0.5 indicate good calibration. Extreme values (near 0 or 1) indicate misfit.

### Prior Analysis

#### plot_prior_predictive
**Use case**: Check if priors produce plausible predictions before fitting.

```python
with model:
    prior_pred = pm.sample_prior_predictive()

# Prior predictive check
az.plot_ppc(prior_pred, group="prior")
```

#### plot_dist
**Use case**: Visualize prior distributions.

```python
# Compare prior and posterior
az.plot_dist(idata.prior["beta"].values.flatten(), label="Prior")
az.plot_dist(idata.posterior["beta"].values.flatten(), label="Posterior")
```

### Model Comparison Plots

#### plot_compare
**Use case**: Visualize model comparison results.

```python
comparison = az.compare({
    "model_a": idata_a,
    "model_b": idata_b,
    "model_c": idata_c,
}, ic="loo")

az.plot_compare(comparison)
```

Shows ELPD differences with standard errors. Models are ranked; overlapping error bars suggest similar predictive performance.

#### plot_elpd
**Use case**: Pointwise ELPD differences between models.

```python
az.plot_elpd({"model_a": idata_a, "model_b": idata_b})
```

Identifies which observations drive model differences.

#### plot_khat (Pareto k diagnostics)
**Use case**: Identify problematic observations for LOO-CV.

```python
az.plot_khat(idata)
```

**Look for**: Most points below 0.7. Points above indicate influential observations where LOO approximation is unreliable.

---

## Divergence Troubleshooting

### Identifying Divergent Regions

```python
# Pair plot with divergences
az.plot_pair(idata, var_names=["alpha", "sigma"], divergences=True)

# Parallel coordinates
az.plot_parallel(idata, var_names=["alpha", "beta", "sigma"])
```

### Common Causes and Fixes

**1. Centered parameterization in hierarchical models**

```python
# BAD: Centered (causes funnel)
alpha = pm.Normal("alpha", mu_alpha, sigma_alpha, dims="group")

# GOOD: Non-centered
alpha_offset = pm.Normal("alpha_offset", 0, 1, dims="group")
alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * alpha_offset, dims="group")
```

**2. Weak priors on scale parameters**

```python
# BAD
sigma = pm.HalfCauchy("sigma", beta=10)

# BETTER
sigma = pm.HalfNormal("sigma", sigma=1)
```

**3. Increase target acceptance**

```python
idata = pm.sample(target_accept=0.95)
```

---

## Model Comparison

### LOO-CV

```python
loo_result = az.loo(idata, pointwise=True)
print(loo_result)

# Pareto k diagnostics
print(f"Good k (< 0.5): {(loo_result.pareto_k < 0.5).sum()}")
print(f"OK k (0.5-0.7): {((loo_result.pareto_k >= 0.5) & (loo_result.pareto_k < 0.7)).sum()}")
print(f"Bad k (> 0.7): {(loo_result.pareto_k > 0.7).sum()}")
```

### WAIC

```python
waic_result = az.waic(idata)
```

### Comparing Models

```python
comparison = az.compare({
    "model_a": idata_a,
    "model_b": idata_b,
}, ic="loo")

print(comparison[["rank", "elpd_loo", "p_loo", "d_loo", "weight", "se", "dse"]])
```

**Key columns**:
- `elpd_loo`: Expected log pointwise predictive density (higher is better)
- `d_loo`: Difference from best model
- `dse`: Standard error of difference
- `weight`: Stacking weight for model averaging

**Decision rule**: If `d_loo` < 2×`dse`, models are effectively indistinguishable.
