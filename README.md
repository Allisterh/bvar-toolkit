# bvar-toolkit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22804032.svg)](https://doi.org/10.5281/zenodo.22804032)

MATLAB code for Bayesian VARs by [Joshua Chan](https://joshuachan.org). The repository archives
fourteen replication packages from [joshuachan.org](https://joshuachan.org/code.html) exactly as
published and factors their shared machinery, from the samplers and shrinkage priors to the
marginal likelihood estimators and forecasting routines, into one tested library.

```matlab
run setup.m                 % adds core/ and third_party/ to the path
cd examples
ex03_minnesota_bvar         % a small BVAR, start to finish
```

The code needs MATLAB with the Statistics and Machine Learning Toolbox. `setup.m` also checks
for the Optimization and System Identification Toolboxes, which a few functions and drivers
use, and names what each missing one would break.

## Learn the Methods

The [tutorials](tutorials/) each answer one research question with the settings of a paper, on
the data of its replication package or of the book, and can be read without MATLAB. Each comes
with a script that runs the same analysis on your own data.

- [Does the Order of the Variables Change My VAR Results?](tutorials/variable_ordering/) Under
  the Cholesky model of stochastic volatility it does: in the 20-variable VAR of Chan, Koop and
  Yu (2024), reversing the order moves the estimated correlation between the PCE inflation and
  PPI finished goods equations from about 0.82 to about 0.19. The order-invariant model gives
  the same estimates in both orders.
- [How Should I Choose the Shrinkage Hyperparameters of My BVAR?](tutorials/shrinkage/) By
  maximizing the marginal likelihood, which the asymmetric conjugate prior of Chan (2022) keeps
  in closed form while shrinking a variable's own lags and other variables' lags by different
  amounts. In a 21-variable VAR, the chosen prior improves the one-quarter-ahead point forecasts
  of 20 of the 21 variables relative to the best symmetric prior.
- [Which Stochastic Volatility Specification Should My VAR Use?](tutorials/sv_specification/)
  The marginal likelihoods of Chan (2023) rank them. In a four-variable quarterly VAR of the
  book's FRED-MD panel, factor stochastic volatility with one factor fits best, ahead of the
  Cholesky specification by 14 and the common volatility by 58, and the shrinkage prior accounts
  for half of the distance between the last two.

The thirteen scripts in [`examples/`](examples/) each run in under a minute and are best read in
order: the building blocks (ex01–ex02, the precision sampler and stochastic volatility), VAR
specifications (ex03–ex08), model comparison and forecasting with an estimated VAR
(ex09–ex10), structural identification by sign restrictions (ex11–ex12), and diagnostics for
the output of a sampler (ex13).
[`examples/README.md`](examples/README.md) lists what each one teaches and which library
functions it calls.

The methods are developed in the book *Bayesian Macroeconometrics: Methods and Applications*
(Chapman & Hall/CRC, forthcoming): see the
[sample chapters](https://joshuachan.org/papers/BayesMacroBook_sample.pdf) and
[its code repository](https://github.com/joshuaccchan/bayesian-macroeconometrics), with
MATLAB, R and Python for all fourteen chapters. [`RELATED.md`](RELATED.md) maps the models here
to the book's R and Python scripts and to the R package `bvars`.

## Reproduce a Paper

Every package is here exactly as published, never edited, under
`replications/<paper>/legacy/`, with a permanent `as-published/<paper>` git tag and the source
zip's md5 recorded in [`provenance.md`](provenance.md). Run those files as you would the
original download. Six packages also have a driver, `run_all.m`, that runs the same computation
with the library functions.

### Which Model Do I Want?

| If you want | Paper | Folder | Driver |
|---|---|---|---|
| Shrinkage priors for a large BVAR (the default choice) | Chan (2021, IJF) | `chan2021_ijf_mahp` | `run_all('MNG',…)` |
| A VAR-SV that does not depend on variable ordering | Chan, Koop & Yu (2024, JBES) | `chan_koop_yu2024_jbes_oisv` | `run_all('OI',…)` |
| Non-Gaussian / serially dependent errors, and marginal likelihoods | Chan (2020, JBES) | `chan2020_jbes_kronecker` | `run_all`, `run_ml` |
| Asymmetric conjugate prior, closed-form ML, sign restrictions | Chan (2022, QE) | `chan2022_qe_acp` | `run_all`, `run_jointden` |
| Which SV specification for a large VAR? | Chan (2023, JoE) | `chan2023_joe_mlvarsv` | `run_all('VAR-SV',…)`, `run_ml` |
| Time-varying parameters, equation by equation | Chan (2023, JBES) | `chan2023_jbes_hybtvp` | `run_all` |
| Which time-varying parameter VAR, by marginal likelihood or DIC | Chan & Eisenstat (2018, JAE) | `chan_eisenstat2018_jae_mltvpsv` | legacy only |
| Forecast comparison across priors and volatility models | Chan (2020, Springer) | `chan2020_springer_largebvar` | legacy only |
| The precision sampler for state space models | Chan & Jeliazkov (2009) | `chan_jeliazkov2009_statespace` | legacy only |
| Sign and ranking restrictions in a large structural VAR | Chan, Matthes & Yu (2026, QE) | `chan_matthes_yu2026_qe_svarsign` | legacy; algorithm in `bvar.structural.sign_assign` |
| Prior sensitivity by automatic differentiation | Chan, Jacobi & Zhu (2019/2020/2022) | `cjz2018_ad_var`, `cjz2019_ad_opthyper`, `cjz2021_jae_ad_ml` | legacy only |

"Legacy only" means the package has no driver yet. The SVAR-sign package will not get one,
because its main programs depend on third-party code and produce figures, so only its algorithm
was extracted. [`tests/golden_runs/manifest.md`](tests/golden_runs/manifest.md) names the few
legacy scripts that do not run as shipped. What to cite for each method, with BibTeX, is in
[`CITING.md`](CITING.md).

## Build on the Code

Across the fourteen packages the same steps of the Gibbs sampler were written out again and
again: the auxiliary mixture sampler for the log-volatility path appears in eight of them, under
three names. The `bvar` package under `core/` has each step once. Call the blocks directly,
start from `bvar.models.var_sv`, the complete sampler for a VAR with stochastic volatility, or
copy the nearest `run_all.m` as a template.

| Namespace | Contents |
|---|---|
| `bvar.priors` | Minnesota, natural conjugate and asymmetric conjugate priors (`minn`, `niw`, `acp_stru`, `acp_redu`), the Minnesota scaling they share (`resid_var_ar4`, `minnesota_C`, `vtheta`), and `acp_opt_kappa`, which chooses the shrinkage hyperparameters by maximizing the closed-form marginal likelihood. |
| `bvar.sv` | Stochastic volatility: the auxiliary mixture sampler for three state equations (`ksc_rw_h0`, `ksc_rw_diffuse`, `ksc_ar1_mean`), a common volatility factor (`csv_armh`), the state-equation parameters (`sv_params`, `sv0_params`) and the Student-t degrees of freedom (`nu_studentt`). |
| `bvar.samplers` | The other Gibbs blocks: VAR coefficients equation by equation (`eq_gauss`, `eq_var_redu_tri`, `eq_svar_oi`, `eq_var_oi`, `eq_tri_cs`, `alp_tri_cs`), factor stochastic volatility (`factor_fsv`, `eq_fsv_load`), the hybrid TVP-VAR (`eq_hyb_tvp`), direct draws under the asymmetric conjugate prior (`acp_theta_sig`) and hierarchical shrinkage (`gig_shrinkage`, `horseshoe_kappa_psi`, `nu_psi_ng`). |
| `bvar.structural` | Impact matrices and identification: the order-invariant impact matrix (`b0_row_sampler`, `construct_Sigt`), the map to the reduced form (`reduced_form`) and sign restrictions (`qr_sign`, `sign_restrict`, `sign_assign`, `irf_redu`). |
| `bvar.ml` | Marginal likelihoods: Chib's method for the models of Chan (2020, JBES) (`kron_bvar*`), adaptive importance sampling for those of Chan (2023, JoE) (`mlvarsv_*`), the closed form under the asymmetric conjugate prior (`acp`), and the integrated likelihoods and log densities they share. |
| `bvar.forecast` | Forecasts from a chain: `iterate` runs one draw forward and scores it, `predictive` returns the mean and standard deviation of the h-step predictive distribution of every draw and `simulate` does the same by simulation when the error covariance varies over time, `tables` accumulates RMSFEs and log predictive likelihoods, and `realtime_loaddata` assembles a real-time data vintage. |
| `bvar.diag` | Diagnostics for MCMC output: inefficiency factors (`inefficiency_factor`), Monte Carlo standard errors (`mcse`) and Geweke's convergence diagnostic (`geweke`), each from the long-run variance that `specvar0` estimates. |
| `bvar.models` | Complete samplers: `var_sv`, a VAR with Cholesky or order-invariant stochastic volatility under the prior of Chan, Koop and Yu (2024), and `var_csv`, a VAR with one common volatility factor. Both can also return their parameter draws. |
| `bvar.util` | Shared pieces: the lag matrix (`build_lags`), the difference matrix of the precision samplers (`diffmat`), sparse expansions (`surform`, `surform2`), credible bands for figures (`shaded_band`), the csv and mat files a script writes at the end (`report`) and small numerical helpers. |

Where two legacy versions of a step differ numerically, both survive under separate names.
[`tests/variant_map.md`](tests/variant_map.md) records which legacy copies each function
stands in for and which pairs must never be merged.

## Verification

Most library functions are extracted from a published package, and a unit test runs the
original code alongside the function and requires identical output, draw for draw under a fixed
seed. Where the library factors a matrix once and the original code factors it twice, the test
first applies the substitutions declared in `tests/unit/private/one_factor_patch.m` to the
original code. Where a function has since been improved, for example with an option in place
of a hard-coded constant, its default still reproduces the published computation and the
improvement has a test of its own. New functions, such as `bvar.models.var_sv`, are tested
against the code they replace. The marginal likelihood code of two packages, Chan (2020, JBES)
and Chan (2023, JoE), contains defects; the library corrects them by default and reproduces the
published computation with `'bugcompat', true`. On every push, CI checks the archived packages
against their `as-published` tags and runs the unit suite, `setup.m`, the examples and the
tutorials' `your_data.m` scripts. To run the tests locally:

```matlab
run tests/unit/run_unit_tests.m
run tests/run_examples.m
```

[`tests/variant_map.md`](tests/variant_map.md) has the full record, including the audits of the
two packages and every deviation from the legacy code, and
[`tests/golden_runs/manifest.md`](tests/golden_runs/manifest.md) lists the captured output of
the original packages in `tests/golden/`.

## Citation

Cite the paper behind each method you use: [`CITING.md`](CITING.md) maps every replication
package, library function and example to its paper, with BibTeX. To cite the toolkit itself:

> Chan, J. C. C. (2026). *bvar-toolkit: MATLAB code for large Bayesian VARs*. Zenodo. https://doi.org/10.5281/zenodo.22804032

The DOI resolves to the latest release. `CITATION.cff` holds the machine-readable record that
GitHub's "Cite this repository" button reads.

## License

MIT; see [`LICENSE`](LICENSE). The license covers the archived packages too: their original
headers, which say "free to use for academic purposes only", are kept verbatim and superseded
by it. Third-party files keep their own licenses, and [`NOTICE.md`](NOTICE.md) lists them.
