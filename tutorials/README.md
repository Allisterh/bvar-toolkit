# Tutorials

Each tutorial addresses one empirical question with the settings of a paper, on the data of its
replication package or of *Bayesian Macroeconometrics*, and summarizes what the paper
contributes. A `build.m` in each folder regenerates every figure and number on the page, and a
`your_data.m` runs the same analysis on other data.

| Tutorial | Code | Paper |
|---|---|---|
| [Does the Order of the Variables Change My VAR Results?](variable_ordering/) | ex06, `bvar.models.var_sv` | Chan, Koop and Yu (2024) |
| [How Should I Choose the Shrinkage Hyperparameters of My BVAR?](shrinkage/) | `bvar.priors.acp_opt_kappa`, `bvar.ml.acp` | Chan (2022) |
| [Which Stochastic Volatility Specification Should My VAR Use?](sv_specification/) | `replications/chan2023_joe_mlvarsv`, `bvar.ml.mlvarsv_*`, `bvar.diag` | Chan (2023a) |
