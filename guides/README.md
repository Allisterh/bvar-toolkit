# How-To Guides

Each guide shows how to use the toolkit's core functions on one small example with generated data.
A guide is a page, `README.md`, and a script, `guide.m`, that prints the numbers and draws the
figures on the page. The guides come in series, and within a series each guide builds on the one
before.

## Getting Started with Bayesian VARs

The guides use the same generated VAR(2) of three variables.

| Guide | What it shows | Core functions |
|---|---|---|
| [How to Stack the Equations of a VAR](var_stacking/) | Stack the equations in matrix form and as one long regression | `bvar.util.build_lags`, `bvar.util.surform2` |
| [How to Draw the VAR Coefficients under the Minnesota Prior](coefficient_draw/) | Get posterior draws of the VAR coefficients | `bvar.priors.minn`, `bvar.samplers.var_coef_joint` |
| [How to Estimate a VAR by Gibbs Sampling](gibbs_sampler/) | Draw the coefficients and the error covariance matrix in turn, and check the chain | `bvar.samplers.var_coef_joint`, `bvar.diag.inefficiency_factor`, `bvar.diag.mcse`, `bvar.diag.geweke` |
| [How to Choose the Shrinkage of a VAR by the Marginal Likelihood](natural_conjugate_ml/) | Choose the shrinkage and the lag length by the closed-form marginal likelihood | `bvar.util.ldet`, `bvar.util.mgammaln`, `bvar.ml.kron_bvar` |

The [examples](../examples/) run the core functions on additional models, with real and generated
data, and the [tutorials](../tutorials/) answer empirical questions with the settings of published
papers.
