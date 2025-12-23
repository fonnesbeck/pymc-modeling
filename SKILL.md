---
name: pymc-modeling
description: >
  Bayesian statistical modeling with PyMC v5+. Use when building probabilistic models,
  specifying priors, running MCMC inference, diagnosing convergence, or comparing models.
  Covers PyMC, ArviZ, pymc-bart, pymc-extras, nutpie, and JAX/NumPyro backends. Triggers
  on tasks involving: Bayesian inference, posterior sampling, hierarchical/multilevel models,
  GLMs, time series, Gaussian processes, BART, mixture models, prior/posterior predictive
  checks, MCMC diagnostics, LOO-CV, WAIC, model comparison, or causal inference with do/observe.
---

# PyMC Modeling

Bayesian modeling workflow for PyMC v5+ with modern API patterns.

**Notebook preference**: Use marimo for interactive modeling unless the project already uses Jupyter.

## Model Specification

### Basic Structure

```python
import pymc as pm
import arviz as az

with pm.Model(coords=coords) as model:
    # Data containers (for out-of-sample prediction)
    x = pm.Data("x", x_obs, dims="obs")

    # Priors
    beta = pm.Normal("beta", mu=0, sigma=1, dims="features")
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Likelihood
    mu = pm.math.dot(x, beta)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs, dims="obs")

    # Inference
    idata = pm.sample()
```

### Coords and Dims

Use coords/dims for interpretable InferenceData when model has meaningful structure:

```python
coords = {
    "obs": np.arange(n_obs),
    "features": ["intercept", "age", "income"],
    "group": group_labels,
}
```

Skip for simple models where overhead exceeds benefit.

### Parameterization

Prefer non-centered parameterization for hierarchical models with weak data:

```python
# Non-centered (better for divergences)
offset = pm.Normal("offset", 0, 1, dims="group")
alpha = mu_alpha + sigma_alpha * offset

# Centered (better with strong data)
alpha = pm.Normal("alpha", mu_alpha, sigma_alpha, dims="group")
```

## Inference

### Default Sampling (nutpie)

Use nutpie as the default sampler—it's Rust-based and typically 2-5x faster:

```python
import nutpie

with model:
    compiled = nutpie.compile_pymc_model(model)
    idata = nutpie.sample(compiled, draws=1000, tune=1000, chains=4, seed=42)
```

### PyMC Native Sampling

Fall back to PyMC's NUTS when nutpie unavailable:

```python
with model:
    idata = pm.sample(draws=1000, tune=1000, chains=4, random_seed=42)
```

### Alternative MCMC Backends

See [references/inference.md](references/inference.md) for:
- **NumPyro/JAX**: GPU acceleration, vectorized chains

### Approximate Inference

For fast (but inexact) posterior approximations:
- **ADVI/DADVI**: Variational inference with Gaussian approximation
- **Pathfinder**: Quasi-Newton optimization for initialization or screening

## Diagnostics

After sampling, always check:

```python
# Summary with convergence diagnostics
az.summary(idata, var_names=["~offset"])  # exclude auxiliary

# Visual diagnostics
az.plot_trace(idata, var_names=["beta", "sigma"])
az.plot_rank(idata)  # rank plots for convergence

# Divergences
idata.sample_stats["diverging"].sum()
```

**Key thresholds:**
- `r_hat < 1.01` (strict) or `< 1.05` (permissive)
- `ess_bulk > 400` and `ess_tail > 400` per chain
- No divergences (or investigate cause)

See [references/diagnostics.md](references/diagnostics.md) for troubleshooting.

## Prior and Posterior Predictive Checks

```python
with model:
    # Prior predictive (before fitting)
    idata.extend(pm.sample_prior_predictive())

    # Posterior predictive (after fitting)
    idata.extend(pm.sample_posterior_predictive(idata))

# Visualize
az.plot_ppc(idata, kind="cumulative")
az.plot_ppc(idata, kind="scatter", flatten=[])
```

## Model Comparison

```python
# Compute LOO-CV (preferred)
az.loo(idata)
az.waic(idata)  # alternative

# Compare models
comparison = az.compare({
    "model_1": idata_1,
    "model_2": idata_2,
}, ic="loo")

az.plot_compare(comparison)
```

Check Pareto k diagnostics: `k > 0.7` indicates problematic observations.

See [references/diagnostics.md](references/diagnostics.md) for handling high Pareto k values.

## Prior Selection

See [references/priors.md](references/priors.md) for:
- Weakly informative defaults by distribution type
- Prior predictive checking workflow
- Domain-specific recommendations

## Common Patterns

### Hierarchical/Multilevel

```python
with pm.Model(coords={"group": groups, "obs": obs_idx}) as hierarchical:
    # Hyperpriors
    mu_alpha = pm.Normal("mu_alpha", 0, 1)
    sigma_alpha = pm.HalfNormal("sigma_alpha", 1)

    # Group-level (non-centered)
    alpha_offset = pm.Normal("alpha_offset", 0, 1, dims="group")
    alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * alpha_offset, dims="group")

    # Likelihood
    y = pm.Normal("y", alpha[group_idx], sigma, observed=y_obs, dims="obs")
```

### GLMs

```python
# Logistic regression
with pm.Model() as logistic:
    beta = pm.Normal("beta", 0, 2.5, dims="features")  # weakly informative
    p = pm.math.sigmoid(pm.math.dot(X, beta))
    y = pm.Bernoulli("y", p=p, observed=y_obs)

# Poisson regression
with pm.Model() as poisson:
    beta = pm.Normal("beta", 0, 1, dims="features")
    mu = pm.math.exp(pm.math.dot(X, beta))
    y = pm.Poisson("y", mu=mu, observed=y_obs)
```

### Gaussian Processes

```python
with pm.Model() as gp_model:
    # Kernel hyperparameters
    ell = pm.InverseGamma("ell", alpha=5, beta=5)
    eta = pm.HalfNormal("eta", sigma=2)

    # Covariance function
    cov = eta**2 * pm.gp.cov.Matern52(1, ls=ell)

    # GP (use HSGP for large datasets)
    gp = pm.gp.Latent(cov_func=cov)
    f = gp.prior("f", X=X)

    # Likelihood
    y = pm.Normal("y", mu=f, sigma=sigma, observed=y_obs)
```

For large datasets, use `pm.gp.HSGP` (Hilbert Space GP approximation).

See [references/gp.md](references/gp.md) for:
- Covariance function selection and combination (additive and multiplicative)
- HSGP configuration (choosing m and L)
- Priors for GP hyperparameters
- Common GP patterns (additive components, heteroscedastic, classification)

### Time Series

```python
with pm.Model(coords={"time": range(T)}) as ar_model:
    rho = pm.Uniform("rho", -1, 1)
    sigma = pm.HalfNormal("sigma", sigma=1)

    y = pm.AR("y", rho=[rho], sigma=sigma, constant=True,
              observed=y_obs, dims="time")
```

See [references/timeseries.md](references/timeseries.md) for:
- Autoregressive models (AR, ARMA)
- Random walk and local level models
- Structural time series (trend + seasonality)
- State space models
- GPs for time series
- Handling multiple seasonalities
- Forecasting patterns

### BART (Bayesian Additive Regression Trees)

```python
import pymc_bart as pmb

with pm.Model() as bart_model:
    mu = pmb.BART("mu", X=X, Y=y, m=50)
    sigma = pm.HalfNormal("sigma", 1)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
```

See [references/bart.md](references/bart.md) for:
- Regression and classification
- Variable importance and partial dependence
- Combining BART with parametric components
- Configuration (number of trees, depth priors)

## Common Pitfalls

See [references/gotchas.md](references/gotchas.md) for:
- Centered vs non-centered parameterization
- Priors on scale parameters
- Label switching in mixtures
- Performance issues (GPs, large Deterministics)

## Causal Inference Operations

### pm.do (Interventions)

Apply do-calculus interventions to set variables to fixed values:

```python
with pm.Model() as causal_model:
    x = pm.Normal("x", 0, 1)
    y = pm.Normal("y", x, 1)
    z = pm.Normal("z", y, 1)

# Intervene: set x = 2 (breaks incoming edges to x)
with pm.do(causal_model, {"x": 2}) as intervention_model:
    idata = pm.sample_prior_predictive()
    # Samples from P(y, z | do(x=2))
```

### pm.observe (Conditioning)

Condition on observed values without intervention:

```python
# Condition: observe y = 1 (doesn't break causal structure)
with pm.observe(causal_model, {"y": 1}) as conditioned_model:
    idata = pm.sample()
    # Samples from P(x, z | y=1)
```

### Combining do and observe

```python
# Intervention + observation for causal queries
with pm.do(causal_model, {"x": 2}) as m1:
    with pm.observe(m1, {"z": 0}) as m2:
        idata = pm.sample()
        # P(y | do(x=2), z=0)
```

## pymc-extras

For specialized models:

```python
import pymc_extras as pmx

# Marginalizing discrete parameters
with pm.Model() as marginal:
    pmx.MarginalMixture(...)

# R2D2 prior for regression
pmx.R2D2M2CP(...)
```
