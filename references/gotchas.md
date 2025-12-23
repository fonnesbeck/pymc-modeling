# Common Pitfalls

## Statistical Issues

### Centered Parameterization with Weak Data

```python
# Causes divergences with few observations per group
alpha = pm.Normal("alpha", mu_alpha, sigma_alpha, dims="group")  # BAD

# Non-centered parameterization
alpha_offset = pm.Normal("alpha_offset", 0, 1, dims="group")
alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * alpha_offset, dims="group")
```

### Flat Priors on Scale Parameters

```python
# Problematic in hierarchical models
sigma = pm.Uniform("sigma", 0, 100)  # BAD
sigma = pm.HalfFlat("sigma")  # BAD

# Weakly informative alternatives
sigma = pm.HalfNormal("sigma", sigma=1)
sigma = pm.HalfCauchy("sigma", beta=1)
sigma = pm.Exponential("sigma", lam=1)
```

### Label Switching in Mixture Models

```python
# Unordered components cause label switching
mu = pm.Normal("mu", 0, 10, dims="component")  # BAD

# Order constraint
mu_raw = pm.Normal("mu_raw", 0, 10, dims="component")
mu = pm.Deterministic("mu", pt.sort(mu_raw), dims="component")
```

### Missing Prior Predictive Checks

Always check prior implications before fitting:

```python
with model:
    prior_pred = pm.sample_prior_predictive()

az.plot_ppc(prior_pred, group="prior")
```

## Performance Issues

### Full GP on Large Datasets

```python
# O(n³) - slow for n > 1000
gp = pm.gp.Marginal(cov_func=cov)
y = gp.marginal_likelihood("y", X=X_large, y=y_obs)

# O(nm) - use HSGP instead
gp = pm.gp.HSGP(m=[30], c=1.5, cov_func=cov)
f = gp.prior("f", X=X_large)
```

### Saving Large Deterministics

```python
# Stores n_obs x n_draws array
mu = pm.Deterministic("mu", X @ beta, dims="obs")  # SLOW

# Don't save intermediate computations
mu = X @ beta  # Not saved, use posterior_predictive if needed
```

### Recompiling for Each Dataset

```python
# Recompiles every iteration
for dataset in datasets:
    with pm.Model() as model:
        # ...
        idata = pm.sample()

# Use pm.Data to avoid recompilation
with pm.Model() as model:
    x = pm.Data("x", x_initial)
    # ...

for dataset in datasets:
    pm.set_data({"x": dataset["x"]})
    idata = pm.sample()
```
