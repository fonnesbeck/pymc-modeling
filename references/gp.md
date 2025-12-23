# Gaussian Processes in PyMC

## Table of Contents
- [GP Fundamentals](#gp-fundamentals)
- [Covariance Functions](#covariance-functions)
- [GP Implementations](#gp-implementations)
- [Scalable GPs (HSGP)](#scalable-gps-hsgp)
- [Priors for GP Hyperparameters](#priors-for-gp-hyperparameters)
- [Common Patterns](#common-patterns)

## GP Fundamentals

A Gaussian process defines a distribution over functions. Key components:
- **Mean function**: Prior expectation (often zero)
- **Covariance function (kernel)**: Encodes smoothness, periodicity, etc.
- **Hyperparameters**: Length scale, amplitude, noise variance

### Basic Structure

```python
import pymc as pm
import numpy as np

with pm.Model() as gp_model:
    # Hyperparameters
    ell = pm.InverseGamma("ell", alpha=5, beta=5)  # length scale
    eta = pm.HalfNormal("eta", sigma=2)            # amplitude
    sigma = pm.HalfNormal("sigma", sigma=1)        # noise

    # Covariance function
    cov = eta**2 * pm.gp.cov.Matern52(1, ls=ell)

    # GP
    gp = pm.gp.Marginal(cov_func=cov)

    # Likelihood (marginalizes out f)
    y_ = gp.marginal_likelihood("y", X=X, y=y, sigma=sigma)
```

## Covariance Functions

### Stationary Kernels

```python
# Squared Exponential (RBF) - infinitely differentiable
cov = pm.gp.cov.ExpQuad(input_dim=1, ls=ell)

# Matern family - finite differentiability
cov = pm.gp.cov.Matern52(input_dim=1, ls=ell)  # twice differentiable (recommended default)
cov = pm.gp.cov.Matern32(input_dim=1, ls=ell)  # once differentiable
cov = pm.gp.cov.Matern12(input_dim=1, ls=ell)  # continuous, not differentiable (Ornstein-Uhlenbeck)

# Rational Quadratic - mixture of RBFs
cov = pm.gp.cov.RatQuad(input_dim=1, ls=ell, alpha=alpha)

# Exponential (same as Matern12)
cov = pm.gp.cov.Exponential(input_dim=1, ls=ell)
```

### Periodic Kernels

```python
# Periodic kernel
cov = pm.gp.cov.Periodic(input_dim=1, period=period, ls=ell)

# Locally periodic (periodic with decay)
cov = pm.gp.cov.Periodic(1, period=period, ls=ell_periodic) * pm.gp.cov.Matern52(1, ls=ell_decay)
```

### Combining Kernels

```python
# Additive kernels (model independent effects)
cov = cov_trend + cov_seasonal + cov_noise

# Product kernels (modulate one by another)
cov = cov_long_term * cov_periodic

# Example: Trend + seasonality
cov_trend = eta_trend**2 * pm.gp.cov.Matern52(1, ls=ell_trend)
cov_seasonal = eta_seasonal**2 * pm.gp.cov.Periodic(1, period=365, ls=ell_seasonal)
cov = cov_trend + cov_seasonal
```

### Multi-dimensional Inputs with ARD

```python
# Automatic Relevance Determination - separate length scale per dimension
ell = pm.InverseGamma("ell", alpha=5, beta=5, dims="features")
cov = pm.gp.cov.Matern52(input_dim=D, ls=ell)
```

## GP Implementations

PyMC offers several GP implementations for different use cases.

### Marginal GP (Default for Regression)

Analytically marginalizes latent function. Best for standard GP regression with Gaussian noise.

```python
with pm.Model() as model:
    cov = eta**2 * pm.gp.cov.Matern52(1, ls=ell)
    gp = pm.gp.Marginal(cov_func=cov)

    # Marginal likelihood integrates out f
    y_ = gp.marginal_likelihood("y", X=X, y=y, sigma=sigma)

    idata = pm.sample()

# Prediction
with model:
    f_star = gp.conditional("f_star", X_new)
    pred = pm.sample_posterior_predictive(idata, var_names=["f_star"])
```

### Latent GP (For Non-Gaussian Likelihoods)

Samples latent function explicitly. Required for classification, count data, etc.

```python
with pm.Model() as model:
    cov = eta**2 * pm.gp.cov.Matern52(1, ls=ell)
    gp = pm.gp.Latent(cov_func=cov)

    # Sample latent function
    f = gp.prior("f", X=X)

    # Non-Gaussian likelihood
    y_ = pm.Bernoulli("y", p=pm.math.sigmoid(f), observed=y)
```

### HSGP (Scalable Approximation)

See [Scalable GPs section](#scalable-gps-hsgp) below.

## Scalable GPs (HSGP)

Hilbert Space Gaussian Process approximation for large datasets. O(nm) instead of O(n³).

### When to Use HSGP

- Dataset size > 500-1000 points
- Stationary kernels (ExpQuad, Matern)
- Low-dimensional inputs (1-3 dimensions typical)

### Basic HSGP Usage

```python
with pm.Model() as model:
    ell = pm.InverseGamma("ell", alpha=5, beta=5)
    eta = pm.HalfNormal("eta", sigma=2)
    sigma = pm.HalfNormal("sigma", sigma=0.5)

    # HSGP approximation
    gp = pm.gp.HSGP(
        m=[20],           # number of basis functions per dimension
        L=[1.3],          # boundary factor (domain scaled)
        cov_func=eta**2 * pm.gp.cov.Matern52(1, ls=ell),
    )

    f = gp.prior("f", X=X)
    y_ = pm.Normal("y", mu=f, sigma=sigma, observed=y)
```

### Choosing m and L

**m (number of basis functions)**:
- Higher m = better approximation, more computation
- Rule of thumb: `m = 20-50` for smooth functions
- Check approximation quality by comparing to exact GP on subset

**L (boundary factor)**:
- Controls domain extension beyond data range
- `L` should be greater than half the data range
- Larger L needed if expecting extrapolation

```python
# Compute L from data range
X_centered = X - X.mean()
X_range = X_centered.max() - X_centered.min()
L = 1.3 * X_range / 2  # extend 30% beyond data

# Or use the c parameter for automatic scaling
gp = pm.gp.HSGP(
    m=[30],
    c=1.5,   # boundary factor multiplier
    cov_func=cov,
)
```

### HSGP for Multiple Dimensions

```python
# 2D input
gp = pm.gp.HSGP(
    m=[15, 15],      # basis functions per dimension
    L=[L_1, L_2],    # boundary factor per dimension
    cov_func=pm.gp.cov.Matern52(2, ls=[ell_1, ell_2]),
)
```

### HSGP with Non-Gaussian Likelihoods

```python
with pm.Model() as model:
    gp = pm.gp.HSGP(m=[25], c=1.5, cov_func=cov)
    f = gp.prior("f", X=X)

    # Classification
    y_ = pm.Bernoulli("y", p=pm.math.sigmoid(f), observed=y)

    # Or Poisson regression
    y_ = pm.Poisson("y", mu=pm.math.exp(f), observed=y)
```

## Priors for GP Hyperparameters

### Length Scale

The length scale controls the "wiggliness" of the function. Smaller = more wiggly.

```python
# InverseGamma - recommended, prevents length scale → 0 or ∞
ell = pm.InverseGamma("ell", alpha=5, beta=5)

# Gamma - if you expect shorter length scales
ell = pm.Gamma("ell", alpha=2, beta=1)

# Log-normal - for order-of-magnitude uncertainty
ell = pm.LogNormal("ell", mu=0, sigma=1)
```

Calibrate to data scale:
```python
# If X spans [0, 10], length scale around 1-5 reasonable
data_range = X.max() - X.min()
ell = pm.InverseGamma("ell", alpha=5, beta=data_range / 2)
```

### Amplitude (eta/marginal standard deviation)

Controls the magnitude of function variation.

```python
# HalfNormal - weakly informative
eta = pm.HalfNormal("eta", sigma=2)

# Based on outcome scale
y_std = y.std()
eta = pm.HalfNormal("eta", sigma=2 * y_std)
```

### Noise (sigma)

Observation noise standard deviation.

```python
# HalfNormal
sigma = pm.HalfNormal("sigma", sigma=1)

# If you have prior knowledge of noise level
sigma = pm.HalfNormal("sigma", sigma=estimated_noise)
```

### Period (for periodic kernels)

```python
# If period known (e.g., yearly = 365 days)
period = 365.0  # fixed

# If period uncertain but approximately known
period = pm.Normal("period", mu=365, sigma=10)
```

## Common Patterns

### GP Regression with Linear Mean Function

```python
with pm.Model() as model:
    # Linear mean
    alpha = pm.Normal("alpha", 0, 10)
    beta = pm.Normal("beta", 0, 1, dims="features")
    mu = alpha + pm.math.dot(X_features, beta)

    # GP for deviations from linear trend
    gp = pm.gp.Marginal(cov_func=cov)
    y_ = gp.marginal_likelihood("y", X=X_gp, y=y, sigma=sigma, mean_func=pm.gp.mean.Constant(c=mu))
```

### Additive GP Components

```python
with pm.Model() as model:
    # Long-term trend
    gp_trend = pm.gp.HSGP(m=[20], c=1.5, cov_func=cov_trend)
    f_trend = gp_trend.prior("f_trend", X=X)

    # Seasonal component
    gp_seasonal = pm.gp.HSGP(m=[30], c=1.5, cov_func=cov_seasonal)
    f_seasonal = gp_seasonal.prior("f_seasonal", X=X)

    # Combined
    f = f_trend + f_seasonal
    y_ = pm.Normal("y", mu=f, sigma=sigma, observed=y)
```

### GP Classification (Binary)

```python
with pm.Model() as model:
    gp = pm.gp.Latent(cov_func=cov)  # or HSGP for large data
    f = gp.prior("f", X=X)

    # Probit link (analytically nicer) or logit
    p = pm.Deterministic("p", pm.math.sigmoid(f))
    y_ = pm.Bernoulli("y", p=p, observed=y)
```

### Heteroscedastic GP (Input-dependent Noise)

```python
with pm.Model() as model:
    # GP for mean
    gp_mean = pm.gp.HSGP(m=[25], c=1.5, cov_func=cov_mean)
    f_mean = gp_mean.prior("f_mean", X=X)

    # GP for log-noise (ensures positivity)
    gp_noise = pm.gp.HSGP(m=[15], c=1.5, cov_func=cov_noise)
    f_noise = gp_noise.prior("f_noise", X=X)

    sigma = pm.math.exp(f_noise)
    y_ = pm.Normal("y", mu=f_mean, sigma=sigma, observed=y)
```

### Prediction with GP

```python
# After sampling
with model:
    # Conditional distribution at new points
    f_star = gp.conditional("f_star", X_new)

    # Include observation noise for predictive distribution
    y_star = pm.Normal("y_star", mu=f_star, sigma=sigma)

    # Sample predictions
    pred = pm.sample_posterior_predictive(idata, var_names=["f_star", "y_star"])
```
