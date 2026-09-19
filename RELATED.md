# Related Implementations

Many of the models in this repository can also be estimated in R and Python, with the code of
the book *Bayesian Macroeconometrics: Methods and Applications* (Chapman & Hall/CRC,
forthcoming). Four of the models of [Chan (2020a)](CITING.md#chan-2020a) are also in the R
package `bvars`.

## The Book's R and Python Code

The [book's repository](https://github.com/joshuaccchan/bayesian-macroeconometrics) has MATLAB,
R and Python code for all fourteen chapters, and the R and Python scripts mirror the MATLAB
ones script for script. They are separate programs from the functions here, with their own
data, prior settings and chain lengths. The table lists them by chapter and notes where a
model differs from its counterpart here.

| Model or method | In this repository | In the book's code |
|---|---|---|
| Precision sampler for state space models | `replications/chan_jeliazkov2009_statespace`, `bvar.util.diffmat`; ex01, a local level model | Chapter 9 ([R][r09], [Python][p09]): `UC_AR`, a local level model with an AR(1) cycle, and `UC_output_gap`, a trend-cycle decomposition of GDP, both drawing the trend with the precision sampler |
| Stochastic volatility by the auxiliary mixture sampler | `bvar.sv.ksc_rw_h0`, `ksc_ar1_mean`; ex02 | Chapter 10 ([R][r10], [Python][p10]): `SVRW`, the same update for random-walk log-volatilities, and `SVRW_YEN`, which estimates that model on yen returns. Chapter 13 ([R][r13], [Python][p13]): `SVAR1`, the same update for stationary AR(1) log-volatilities |
| Unobserved components model with stochastic volatility | `UC.m` in `replications/chan_jeliazkov2009_statespace`, with stochastic volatility in the transitory component only | Chapter 10 ([R][r10], [Python][p10]): `UCSV`, the model of Stock and Watson (2007), with stochastic volatility in both the trend and the transitory component |
| Minnesota and natural conjugate priors | `bvar.priors.minn`, `bvar.priors.niw`; ex03 | Chapter 12 ([R][r12], [Python][p12]): `Minn_NCP`, the natural conjugate prior elicited Minnesota-style, and `VAR_NCP`, which forecasts with it and computes the marginal likelihood. Chapter 13 ([R][r13], [Python][p13]): `Minn_indep`, the independent Minnesota prior. Chapter 14 ([R][r14], [Python][p14]): `pred_largeVAR_NCP` |
| Asymmetric conjugate prior | `bvar.priors.acp_stru`, `acp_redu`, `acp_opt_kappa`, `bvar.ml.acp`; `replications/chan2022_qe_acp` | Chapter 12 ([R][r12], [Python][p12]): `ml_VAR_ACP`, the closed-form marginal likelihood, and `VAR_ACP_kappa`, which chooses the shrinkage hyperparameters by maximizing it over a grid |
| Common stochastic volatility | `bvar.sv.csv_armh`; ex05; model 3 of `replications/chan2020_jbes_kronecker` | Chapter 14 ([R][r14], [Python][p14]): `pred_largeVAR_CSV`, the VAR with common stochastic volatility under a natural conjugate prior, and `sample_CSV_h_ARMH`, the ARMH step for the volatility path that `csv_armh` implements |
| Cholesky stochastic volatility | `bvar.models.var_sv` with `'model'` set to `'CS'`, `bvar.samplers.eq_tri_cs`, `alp_tri_cs`; ex04 | Chapter 14 ([R][r14], [Python][p14]): `pred_largeVAR_SV`, with stationary AR(1) log-volatilities, as here. Chapter 13 ([R][r13], [Python][p13]): `pred_VAR_SV`, with random-walk log-volatilities |
| Order-invariant stochastic volatility | `bvar.models.var_sv`, `bvar.structural.b0_row_sampler`, `bvar.samplers.eq_svar_oi`, `eq_var_oi`; ex06; `replications/chan_koop_yu2024_jbes_oisv` | Chapter 13 ([R][r13], [Python][p13]): `pred_VAR_OISV`; `sample_B0`, the row-by-row draw of the impact matrix that `b0_row_sampler` implements; and `compare_VAR_SV`, which compares the forecasts of the homoskedastic, Cholesky and order-invariant models. Chapter 14 ([R][r14], [Python][p14]): `pred_largeVAR_OISV` |
| Factor stochastic volatility | `bvar.samplers.factor_fsv`, `eq_fsv_load`; ex07 | Chapter 14 ([R][r14], [Python][p14]): `pred_largeVAR_FSV`. It leaves the loadings unrestricted and gives the factor log-volatilities zero means, where ex07 sets the diagonal of the loadings to one, with zeros above it, and gives the factor log-volatilities free means; the covariance matrix is the same either way |
| Recursive forecast evaluation | `bvar.forecast.iterate`, `bvar.forecast.tables`; ex10 | Chapter 14 ([R][r14], [Python][p14]): `forecast_largeVAR`, which compares eight specifications of a 25-variable VAR |

The Chapter 14 samplers `pred_largeVAR_SV` and `pred_largeVAR_OISV` take the prior on the VAR
coefficients as an argument. Their option `'mahp'` has the form of the prior in
`bvar.models.var_sv`, a Minnesota-type prior with half-Cauchy global and local scales. The
Minnesota-type normal-gamma prior of [Chan (2021)](CITING.md#chan-2021), used in ex10 and
`replications/chan2021_ijf_mahp`, is not among the options.

The book's code has no counterpart of the hybrid TVP-VAR of ex08, which is in structural form,
while Chapter 13 treats the TVP-VAR in reduced form. Nor does it have the sign-restriction
algorithms of ex11, ex12 and `replications/chan_matthes_yu2026_qe_svarsign`, or the models
of the three packages that use automatic differentiation (`replications/cjz*`).

## The R Package bvars

[`bvars`](https://cran.r-project.org/package=bvars), by Rui Liu, Andrés Ramirez Hassan and
Tomasz Woźniak, is on CRAN. Its description lists the Bayesian VAR of Chan (2020a) with a
Minnesota prior and normal or Student-t errors, homoskedastic or with a common stochastic
volatility, centered or non-centered, together with the model of
[Shang, Wang and Woźniak (2026)](https://doi.org/10.48550/arXiv.2608.28087). These are models
1, 2, 3 and 5 of `replications/chan2020_jbes_kronecker`: BVAR, BVAR-t, BVAR-CSV and
BVAR-t-CSV. The moving average errors of models 4 and 6 to 8 are not in the description. The
package also computes density forecasts and forecast error variance decompositions, and it
shares its objects and workflow with `bsvars`, `bsvarSIGNs` and `bpvars`; see
[bsvars.org/bvars](https://bsvars.org/bvars/).

[r09]: https://github.com/joshuaccchan/bayesian-macroeconometrics/tree/main/code/R/chapter09
[p09]: https://github.com/joshuaccchan/bayesian-macroeconometrics/tree/main/code/python/chapter09
[r10]: https://github.com/joshuaccchan/bayesian-macroeconometrics/tree/main/code/R/chapter10
[p10]: https://github.com/joshuaccchan/bayesian-macroeconometrics/tree/main/code/python/chapter10
[r12]: https://github.com/joshuaccchan/bayesian-macroeconometrics/tree/main/code/R/chapter12
[p12]: https://github.com/joshuaccchan/bayesian-macroeconometrics/tree/main/code/python/chapter12
[r13]: https://github.com/joshuaccchan/bayesian-macroeconometrics/tree/main/code/R/chapter13
[p13]: https://github.com/joshuaccchan/bayesian-macroeconometrics/tree/main/code/python/chapter13
[r14]: https://github.com/joshuaccchan/bayesian-macroeconometrics/tree/main/code/R/chapter14
[p14]: https://github.com/joshuaccchan/bayesian-macroeconometrics/tree/main/code/python/chapter14
