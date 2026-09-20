# Tutorials

Each tutorial addresses one empirical question with the settings of a paper, on the data of its
replication package or of *Bayesian Macroeconometrics*, and summarizes what the paper
contributes. A `build.m` in each folder regenerates every figure and number on the page, and a
`your_data.m` runs the same analysis on other data.

The paper column links to the published version. [`CITING.md`](../CITING.md) gives the full
reference for each, with a working-paper copy where one is available, the replication package it
came from, and a BibTeX entry.

| Tutorial | Code | Paper |
|---|---|---|
| [Does the Order of the Variables Change My VAR Results?](variable_ordering/) | ex06, `bvar.models.var_sv` | [Chan, Koop and Yu (2024)](https://doi.org/10.1080/07350015.2023.2252039) |
| [How Should I Choose the Shrinkage Hyperparameters of My BVAR?](shrinkage/) | `bvar.priors.acp_opt_kappa`, `bvar.ml.acp` | [Chan (2022)](https://doi.org/10.3982/QE1381) |
| [Which Stochastic Volatility Specification Should My VAR Use?](sv_specification/) | `replications/chan2023_joe_mlvarsv`, `bvar.ml.mlvarsv_*`, `bvar.diag` | [Chan (2023a)](https://doi.org/10.1016/j.jeconom.2022.11.003) |
| [Does Modeling the Volatility Improve My Forecasts?](forecasting/) | `bvar.models.var_csv`, `bvar.models.var_sv`, `bvar.forecast.simulate` | [Carriero, Clark and Marcellino (2016)](https://doi.org/10.1080/07350015.2015.1040116); [Chan, Koop and Yu (2024)](https://doi.org/10.1080/07350015.2023.2252039) |
