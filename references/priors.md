# Prior Selection Guide

## Table of Contents
- [Philosophy](#philosophy)
- [Weakly Informative Defaults](#weakly-informative-defaults)
- [Prior Predictive Checking](#prior-predictive-checking)
- [Regularizing Priors](#regularizing-priors)
- [Domain-Specific Guidance](#domain-specific-guidance)

## Philosophy

Priors encode information before seeing data. The goal is not "non-informative" priors (which don't exist) but **weakly informative** priors that:
1. Rule out implausible parameter values
2. Keep probability mass in reasonable ranges
3. Don't overwhelm the likelihood with strong prior information

Always perform prior predictive checks to validate prior choices.

## Weakly Informative Defaults

### Location Parameters (means, intercepts)

```python
# Standardize predictors first, then:
beta = pm.Normal("beta", mu=0, sigma=1)       # coefficients on standardized scale
intercept = pm.Normal("intercept", mu=0, sigma=10)  # if outcome not standardized
```

For unstandardized data, center prior on domain-reasonable value with SD covering plausible range.

### Scale Parameters (standard deviations, variances)

```python
# Half-Normal (recommended default)
sigma = pm.HalfNormal("sigma", sigma=1)

# Half-Cauchy (heavier tails, more robust)
sigma = pm.HalfCauchy("sigma", beta=1)

# Exponential (strong regularization toward 0)
sigma = pm.Exponential("sigma", lam=1)

# InverseGamma (for variance, not SD)
variance = pm.InverseGamma("variance", alpha=2, beta=1)
```

**Avoid**: `pm.Uniform(0, large_number)` and `pm.HalfFlat()` in hierarchical models.

### Correlation Matrices

```python
# LKJ prior on correlation matrix
# eta=1: uniform over valid correlations
# eta>1: concentrates toward identity (less correlation)
# eta<1: concentrates toward extreme correlations
chol, corr, stds = pm.LKJCholeskyCov(
    "chol", n=n_dims, eta=2.0,
    sd_dist=pm.HalfNormal.dist(sigma=1)
)
```

### Probability Parameters

```python
# Beta prior for bounded [0, 1]
p = pm.Beta("p", alpha=1, beta=1)           # uniform
p = pm.Beta("p", alpha=2, beta=2)           # slight mode at 0.5
p = pm.Beta("p", alpha=0.5, beta=0.5)       # Jeffreys, concentrates at extremes

# Logit-normal for softer constraints
logit_p = pm.Normal("logit_p", mu=0, sigma=1.5)
p = pm.math.sigmoid(logit_p)
```

### Count/Rate Parameters

```python
# Gamma for rates (positive, right-skewed)
rate = pm.Gamma("rate", alpha=2, beta=1)

# Log-normal when multiplicative effects expected
rate = pm.LogNormal("rate", mu=0, sigma=1)
```

### Regression Coefficients

```python
# Standard: Normal centered at 0
beta = pm.Normal("beta", mu=0, sigma=2.5)   # logistic regression default

# Heavy-tailed: Student-t for robustness to outlier predictors
beta = pm.StudentT("beta", nu=3, mu=0, sigma=2.5)

# Sparse: Horseshoe prior (pymc-extras)
import pymc_extras as pmx
beta = pmx.Horseshoe("beta", dims="features")

# R2D2 prior (induced prior on R-squared)
beta = pmx.R2D2M2CP("beta", dims="features", r2=0.5)
```

## Prior Predictive Checking

Always simulate from priors before fitting:

```python
with model:
    prior_pred = pm.sample_prior_predictive(samples=500)

# Check if prior predictions span reasonable outcome range
import arviz as az
az.plot_ppc(prior_pred, group="prior", kind="cumulative")

# Numerical summary
prior_y = prior_pred.prior_predictive["y"].values.flatten()
print(f"Prior predictive range: [{prior_y.min():.2f}, {prior_y.max():.2f}]")
print(f"Prior predictive mean: {prior_y.mean():.2f}")
```

**Warning signs**:
- Prior predictive covers implausible values (negative counts, >100% probabilities)
- Prior predictive too concentrated (won't explore parameter space)
- Prior predictive extremely wide (suggests weak identifiability)

## Regularizing Priors

### For High-Dimensional Problems

```python
# Horseshoe for sparse signals
tau = pm.HalfCauchy("tau", beta=1)  # global shrinkage
lam = pm.HalfCauchy("lam", beta=1, dims="features")  # local shrinkage
beta = pm.Normal("beta", mu=0, sigma=tau * lam, dims="features")

# Regularized Horseshoe (finite variance)
import pymc_extras as pmx
beta = pmx.R2D2M2CP("beta", dims="features", r2=0.5, r2_std=0.2)
```

### For Hierarchical Models

```python
# Hyperprior on group-level SD controls pooling
# Tight prior = more pooling toward grand mean
# Diffuse prior = less pooling, closer to no-pooling
sigma_group = pm.HalfNormal("sigma_group", sigma=0.5)  # moderate pooling
```

## Domain-Specific Guidance

### Biostatistics/Epidemiology

- Log-odds ratios: `pm.Normal(0, 2.5)` allows ORs from ~0.01 to ~100
- Hazard ratios: similar, `pm.Normal(0, 1)` on log scale
- Prevalence: `pm.Beta(1, 1)` or `pm.Beta(2, 2)` unless strong prior info

### Economics/Social Science

- Elasticities: `pm.Normal(0, 1)` on log-log models
- Treatment effects: center on 0, SD based on plausible effect sizes
- Time trends: `pm.Normal(0, 0.1)` for small per-period changes

### Physical Sciences

- Incorporate physical constraints (positivity, conservation laws)
- Use informative priors from previous experiments when available
- Consider measurement error models for covariates

### Machine Learning / Prediction

- Focus on predictive performance, less on parameter interpretation
- Horseshoe or R2D2 for automatic relevance determination
- Cross-validate prior choices via LOO-CV
