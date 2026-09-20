# Which Stochastic Volatility Specification Should My VAR Use?

*Code: [`build.m`](build.m) and [`your_data.m`](your_data.m), which call
[`run_ml`](../../replications/chan2023_joe_mlvarsv/run_ml.m) of the Chan (2023) replication
package and the estimators in [`core/+bvar/+ml`](../../core/+bvar/+ml). Method:
[Chan (2023)](../../CITING.md#chan-2023a).*

In this tutorial we compare five ways of modeling the error covariance matrix of a Bayesian VAR
by their marginal likelihoods: constant volatility, one common volatility factor (VAR-CSV), one
volatility process per equation (VAR-SV), a few volatility factors (VAR-FSV), and one volatility
process per equation with an outlier component (VAR-SVO). Chan (2023) estimates these marginal
likelihoods by integrating out the VAR coefficients analytically and the log-volatilities by
adaptive importance sampling. In a four-variable quarterly VAR of US industrial production, the
unemployment rate, PCE prices and the federal funds rate from 1961 to 2025, every specification
with stochastic volatility beats the homoskedastic VAR by more than 300 in log marginal
likelihood. Factor stochastic volatility with one factor fits best at 477.1, ahead of the
Cholesky specification at 463.0 and the common volatility at 418.9. Adding the outlier component
to the Cholesky specification changes nothing (463.5 against 463.0, with numerical standard
errors of 0.25 and 0.30), even though it marks 2020Q2 and 2020Q3 as outliers with probability
one. Half of the distance between the Cholesky and the common specification comes from the prior:
restricted to one shrinkage hyperparameter for own and other lags, VAR-SV falls to 441.2.

![Common volatility and outlier probabilities](fig_outliers.png)

*Figure 1: The posterior mean of the common volatility $`\mathrm{e}^{h_t/2}`$ under VAR-CSV (top)
and the posterior probability that a quarter is an outlier under VAR-SVO (bottom), 1961Q2 to
2025Q3. The common volatility peaks in 1975, 1980, 2008 and 2020; the outlier probability passes
one half only in 2020.*

## The Five Specifications

All five models have the same conditional mean, a reduced-form VAR with $`p`$ lags,

$$\mathbf{y}_t = \mathbf{a}_0 + A_1\mathbf{y}_{t-1} + \cdots + A_p\mathbf{y}_{t-p} + \boldsymbol{\varepsilon}_t, \qquad \boldsymbol{\varepsilon}_t \sim N(\mathbf{0}, \Sigma_t),$$

and differ in the covariance matrix $`\Sigma_t`$ of the innovations and in the prior on the VAR
coefficients. Write $`D_t = \mathrm{diag}(\mathrm{e}^{h_{1t}}, \ldots, \mathrm{e}^{h_{nt}})`$ for a diagonal matrix
of volatilities.

| Model | Error covariance | Volatility processes | Shrinkage hyperparameters |
|---|---|---|---|
| VAR | $`\Sigma`$ | none | one, fixed |
| VAR-CSV | $`\mathrm{e}^{h_t}\Sigma`$ | 1 | one, estimated |
| VAR-SV | $`\Sigma_t^{-1} = B_0'D_t^{-1}B_0`$ | $`n`$ | own lags, other lags, impact matrix |
| VAR-FSV | $`\Sigma_t = LG_tL' + D_t`$ | $`n + r`$ | own lags, other lags |
| VAR-SVO | $`\Sigma_t^{-1} = o_t^{-2}B_0'D_t^{-1}B_0`$ | $`n`$, and an outlier scale | own lags, other lags, impact matrix |

The homoskedastic VAR holds $`\Sigma_t = \Sigma`$ over the whole sample. It has a natural
conjugate prior on the coefficients and the covariance matrix, which makes its marginal
likelihood available in closed form.

VAR-CSV (Carriero, Clark and Marcellino, 2016) scales one covariance matrix by a common
volatility, $`\Sigma_t = \mathrm{e}^{h_t}\Sigma`$, where the log-volatility follows a stationary AR(1)
process $`h_t = \phi h_{t-1} + u_t^h`$ with $`u_t^h \sim N(0,\sigma^2)`$. Its unconditional mean is
zero for identification, since the level of the volatility is absorbed by $`\Sigma`$. All error
variances then move in proportion, and the correlations are constant. The natural conjugate
prior keeps estimation fast even for large $`n`$.

VAR-SV, the Cholesky stochastic volatility of Cogley and Sargent (2005) and Carriero, Clark and
Marcellino (2019), gives each equation its own volatility: $`\Sigma_t^{-1} = B_0'D_t^{-1}B_0`$,
where $`B_0`$ is lower triangular with ones on the diagonal, and each $`h_{it}`$ follows an AR(1)
process with its own mean, persistence and variance. The $`n`$ volatilities let the variances and
the correlations move separately, at the cost of a prior that depends on the order of the
variables ([tutorial 1](../variable_ordering/) measures that dependence).

VAR-FSV takes the innovations to load on $`r`$ latent factors, $`\boldsymbol{\varepsilon}_t =
L\mathbf{f}_t + \mathbf{u}_t`$, where $`\mathbf{u}_t \sim N(\mathbf{0}, D_t)`$ and
$`\mathbf{f}_t \sim N(\mathbf{0}, G_t)`$ are independent, $`L`$ is $`n\times r`$ and lower
triangular with ones on the diagonal, and $`G_t`$ collects $`r`$ further volatilities. The
covariance $`\Sigma_t = LG_tL' + D_t`$ is driven by $`n + r`$ volatility processes, and the
identification restrictions again depend on the order of the variables.

VAR-SVO extends VAR-SV with the outlier component of Stock and Watson (2016), which Carriero,
Clark, Marcellino and Mertens (2024) use for the pandemic observations. A scale $`o_t`$ multiplies
the whole covariance matrix: $`o_t = 1`$ with probability $`1 - p_o`$, and with probability $`p_o`$
it is drawn from a uniform distribution on $`(2, 20)`$, discretized to 31 points. The outlier
probability has a $`B(2.5, 37.5)`$ prior, calibrated to one outlier every four years in quarterly
data. Section 14.2 of *Bayesian Macroeconometrics* gives the same component to VAR-CSV.

The priors on the VAR coefficients differ with the specification, and the difference matters for
the comparison. The VAR and VAR-CSV use the natural conjugate prior, whose Kronecker structure
gives the analytical results that make them fast, and which shrinks the coefficients on own and
other lags by the same amount. The other three specifications have independent priors across
equations, which allows two shrinkage hyperparameters, one for own lags and one for lags of other
variables. Every shrinkage hyperparameter except the one of the homoskedastic VAR is estimated
with the model, under a gamma prior. We return to this difference below.

## How the Marginal Likelihood Is Estimated

The marginal likelihood is the density of the data under a model with all parameters integrated
out. For these models it is a high-dimensional integral over the VAR coefficients, the
log-volatilities, the latent factors and the remaining parameters, and Chan (2023) estimates it
by combining two variance-reduction techniques.

The first is conditional Monte Carlo. Given the log-volatilities and the parameters of the
covariance structure, the VAR coefficients enter a linear regression with a conjugate prior, and
for VAR-FSV the latent factors are Gaussian, so both can be integrated out analytically. What
remains is an integrated likelihood $`p(\mathbf{y}\mid\mathbf{h},\theta)`$ that can be evaluated
without simulating the VAR coefficients, which removes their contribution to the variance of the
estimate. In a large system they are the bulk of the parameters, so that contribution is the
bulk of the variance.

The second is adaptive importance sampling. The log-volatilities and the remaining parameters are
drawn from an importance density fitted to the posterior draws by the cross-entropy method of
Chan and Eisenstat (2015): a Gaussian density for each log-volatility path, parameterized as an
AR(1) process with time-varying intercepts and variances, and normal, truncated normal and gamma
densities for the other blocks. The estimate is the average of $`M`$ importance weights, and its
numerical standard error comes from 50 batches of weights. Sections 5.1.4 and 11.1.3 of
*Bayesian Macroeconometrics* apply the same two ideas to an autoregression with $`t`$ errors and
to a static factor model.

## Results for a Four-Variable VAR

We take four of the 25 series of the panel in Section 14.2 of *Bayesian Macroeconometrics*:
industrial production, the unemployment rate, the PCE price index and the federal funds rate,
transformed to stationarity with the codes of FRED-MD (McCracken and Ng, 2016). The file in this
folder holds all 25 of them, so other selections need only a change of names in the settings
block. The log-differenced series are multiplied by 100, and the two series already in percentage
points are left as they are, as in the book's `forecast_largeVAR.m`. We then average the three
months of each quarter, which gives 266 quarters from 1959Q2 to 2025Q3. The VAR has four lags and
the first eight quarters serve as initial conditions, so the estimation sample runs from 1961Q2
to 2025Q3, with $`T = 258`$.

The priors, the lag length and the sampler settings are those of Section 5 of Chan (2023): the
shrinkage hyperparameters are estimated with the model, each chain keeps 20,000 draws after a
burn-in of 1,000, and each marginal likelihood uses an importance sample of 10,000 draws. Four
variables keep the whole comparison under ten minutes; the cost of a run grows quickly with the
number of variables, since the importance sampler factorizes an $`nk \times nk`$ matrix for every
draw, with $`k = np + 1`$.

*Table 1: Log marginal likelihoods of the four-variable VAR, with numerical standard errors. The
last column is the difference from the best specification. VAR-NCP has no standard error: its
marginal likelihood is available in closed form.*

| Model | Log marginal likelihood | Numerical standard error | Difference |
|---|---|---|---|
| VAR-NCP | 99.9 | | -377.2 |
| VAR-CSV | 418.9 | 0.11 | -58.1 |
| VAR-SV | 463.0 | 0.30 | -14.0 |
| VAR-FSV, $`r = 1`$ | 477.1 | 0.55 | |
| VAR-FSV, $`r = 2`$ | 469.0 | 0.45 | -8.1 |
| VAR-FSV, $`r = 3`$ | 456.9 | 0.56 | -20.1 |
| VAR-SVO | 463.5 | 0.25 | -13.5 |

We note three results in the table. First, time-varying volatility matters far more than the
choice among its forms: the homoskedastic VAR trails the common-volatility model by 319 and the
best specification by 377. Second, the flexible specifications beat the common volatility by 44
to 58, so one volatility process is too restrictive for these four series. Third, the outlier
component leaves the fit where it was: VAR-SVO and VAR-SV differ by 0.5, against numerical
standard errors of 0.25 and 0.30.

The marginal likelihood also selects the number of factors. One factor fits best, and each
further factor costs 8 to 12, which is the price of the extra loadings and the extra
log-volatility process. At four variables only one factor satisfies the identification condition
$`r \leqslant (n-1)/2`$ of Anderson and Rubin (1956) that Chan (2023) adopts, so we report the
values at $`r = 2`$ and $`r = 3`$ for comparison only. In the larger systems of Chan (2023) the
preferred number of factors rises with the dimension.

## Is It the Volatility or the Prior?

The five specifications differ in their priors as well as their likelihoods, and the two effects
can be separated. VAR-SV, VAR-FSV and VAR-SVO allow two shrinkage hyperparameters, one for own
lags and one for the lags of other variables, while the natural conjugate prior of the
homoskedastic VAR and VAR-CSV has a single one. Restricting VAR-SV to a single hyperparameter
gives the symmetric prior in Table 2.

*Table 2: Posterior means and standard deviations of the shrinkage hyperparameters. The
symmetric prior restricts the own-lag and other-lag hyperparameters to be equal, and VAR-CSV has
one hyperparameter for all lags.*

| Model | Log marginal likelihood | Own lags | Other lags | Impact matrix |
|---|---|---|---|---|
| VAR-CSV | 418.9 | 0.118 (0.04) | 0.118 (0.04) | |
| VAR-SV, symmetric prior | 441.2 | 0.0281 (0.008) | 0.0281 (0.008) | 0.0421 (0.061) |
| VAR-SV | 463.0 | 0.109 (0.039) | 0.00231 (0.0015) | 0.0367 (0.051) |
| VAR-FSV, $`r = 1`$ | 477.1 | 0.1 (0.037) | 0.00132 (0.0011) | |
| VAR-FSV, $`r = 2`$ | 469.0 | 0.0998 (0.037) | 0.00185 (0.0014) | |
| VAR-FSV, $`r = 3`$ | 456.9 | 0.0958 (0.036) | 0.00207 (0.0015) | |
| VAR-SVO | 463.5 | 0.101 (0.036) | 0.00295 (0.0018) | 0.0448 (0.062) |

The restriction costs VAR-SV 21.8 in log marginal likelihood, and the estimated hyperparameters
show why: the coefficients on other variables' lags are shrunk about fifty times harder than
those on own lags, a difference the symmetric prior cannot express. The remaining 22.3 of the
44.1 that separates VAR-SV from VAR-CSV comes from the likelihood, since VAR-SV under the
symmetric prior still beats VAR-CSV by that much. At four variables the prior and the volatility
specification therefore contribute about equally. In the 15-variable system of Chan (2023), where
the number of cross-variable coefficients is far larger, the prior accounts for nearly all of the
difference, and VAR-SV under the symmetric prior fits slightly worse than VAR-CSV (Table 3 of the
paper).

## What the Outlier Component Picks Up

The outlier component gives each quarter a scale $`o_t`$ that multiplies the whole covariance
matrix, with a prior frequency of one outlier every four years. The data put its posterior mean
at 0.022, a third of the prior mean, which amounts to 4.2 outlier quarters in 258. Only two
quarters have a posterior outlier probability above one half, and both are unambiguous: 2020Q2
and 2020Q3, with probability 1.00 and posterior means of $`o_t`$ of 10.8 and 9.2. The next
largest probability is 0.49, in 1980Q2. Figure 1 shows the whole path beside the common
volatility of VAR-CSV, which rises in 1975, 1980, 2008 and 2020 and reaches 8.6 in 2020Q2 against
0.65 at its quietest in 1964Q1.

The outlier component absorbs the pandemic quarters that would otherwise lift the persistent
volatility, which is what Carriero, Clark, Marcellino and Mertens (2024) designed it for, and
Section 14.2 of *Bayesian Macroeconometrics* fits the same component to the common volatility of
the 25-variable monthly panel. Here it leaves the marginal likelihood where it was, so on this
sample the Cholesky volatilities already account for the pandemic.

## Checking the Estimates

Two numbers decide whether the ranking in Table 1 can be read at face value. The first is the
numerical standard error of each estimate, which comes from 50 batches of importance weights.
Those run from 0.11 to 0.56, against differences of 8.1 between the best two specifications and
14.0 between the best and VAR-SV, so the ordering is not an artifact of simulation error. The
importance sampling estimator stays consistent however well or badly the importance density fits;
a poor fit shows up in the standard error rather than in the estimate.

The second is how well the chains that fit those importance densities mix. The table gives
inefficiency factors at a truncation lag of 200, computed with `bvar.diag`, over the
hyperparameters, and the count of parameters whose Geweke statistic rejects at the 5% level.

*Table 3: Inefficiency factors by parameter group, and the number of parameters whose Geweke
statistic rejects at the 5% level, from 20,000 draws.*

| Model | $`\kappa`$ | $`\mu`$ | $`\phi`$ | $`\sigma^2`$ | Geweke rejections |
|---|---|---|---|---|---|
| VAR-CSV | 47 | | 25 | 31 | 2 of 3 |
| VAR-SV | 1 to 22 | 1 to 2 | 6 to 15 | 22 to 45 | 4 of 15 |
| VAR-SV, symmetric prior | 1 to 5 | 1 to 2 | 7 to 14 | 23 to 41 | 8 of 15 |
| VAR-FSV, $`r = 1`$ | 3 to 44 | 2 to 13 | 9 to 80 | 34 to 105 | 9 of 17 |
| VAR-FSV, $`r = 2`$ | 4 to 40 | 1 to 17 | 7 to 16 | 22 to 65 | 8 of 20 |
| VAR-FSV, $`r = 3`$ | 4 to 34 | 1 to 17 | 7 to 37 | 26 to 102 | 4 of 23 |
| VAR-SVO | 2 to 18 | 1 to 3 | 7 to 18 | 26 to 58 | 6 of 15 |

The volatility parameters are the slow ones. An inefficiency factor of 100 means that 20,000
draws carry the information of 200 independent ones, which suffices for fitting the importance
density and is thin for a posterior standard deviation of $`\sigma_i^2`$. Geweke's statistic
rejects for between a quarter and a half of the hyperparameters in each model, and the
log-volatility variances account for 16 of the 31 rejections with the persistences for another
11, so a chain several times longer is worth running before quoting posterior summaries of those
two blocks. Section 6.5 of *Bayesian Macroeconometrics* sets out all three diagnostics,
and [ex13](../../examples/ex13_mcmc_diagnostics.m) applies them to a VAR-SV sampler in detail.

## Applying the Method to Your Data

The script [`your_data.m`](your_data.m) runs the same comparison on any dataset. Set the file,
the columns by name, which of them are already in percentage points, the date column, the numbers
of factors and the chain lengths in its settings block. It drops rows missing at either end of
the sample, and stops on a missing value inside it, on months that are not consecutive and on a
constant series. With its defaults it compares seven specifications on the four series of this
page with short chains, in well under a minute, and prints the log marginal likelihoods and the
quarters the outlier component picks out. One call estimates a model and its marginal likelihood:

```matlab
out = run_ml('VAR-SV', false, false, nsim, burnin, seed, [], 'data', data, 'r', r, 'M', M);
```

Here `data` holds the series in columns, oldest first, with the first eight rows as initial
conditions; `out.lml` and `out.lmlstd` are the log marginal likelihood and its numerical standard
error. The second and third arguments hold the shrinkage hyperparameters fixed and impose the
symmetric prior, and `'r'` sets the number of factors of VAR-FSV. The lag length is four and the
first eight rows are the initial conditions, as in Chan (2023), so data of another frequency
should be aggregated to quarters, as `your_data.m` does with its `rows = "months"` setting.

## Implementations in R and Python

The book repository
[bayesian-macroeconometrics](https://github.com/joshuaccchan/bayesian-macroeconometrics)
implements the samplers of these models in MATLAB, R and Python: `chapter14/pred_largeVAR_CSV`,
`pred_largeVAR_SV` and `pred_largeVAR_FSV` for the three stochastic volatility specifications,
`chapter14/VAR_CSV_o` for a common volatility with an outlier component, and
`chapter11/SFM_CE` for a marginal likelihood that combines an integrated likelihood with the
cross-entropy method. [`RELATED.md`](../../RELATED.md) lists the R and Python counterparts of the
other models in this repository.

## Reproducing the Results

To reproduce all results on this page, run the build script from the root of the repository:

```matlab
run tutorials/sv_specification/build.m
```

The eight runs are independent, and `build.m` saves each one in a `runs` folder next to itself,
so an interrupted build resumes where it stopped. Setting `build_runs` to a subset of 1:8 before
running the script computes only those runs, which spreads them over several MATLAB sessions.
The computation took 6.6 minutes using MATLAB R2025b on a computer with an Intel Core Ultra 7
255U processor and 32 GB of RAM. All results in this tutorial are printed in
[`build_log.txt`](build_log.txt), and the figure is saved in the same folder.

## References

Anderson, T. W. and Rubin, H. (1956). Statistical Inference in Factor Analysis. In:
*Proceedings of the Third Berkeley Symposium on Mathematical Statistics and Probability*,
Volume 5, 111-150. University of California Press.

Carriero, A., Clark, T. E. and Marcellino, M. (2016). Common Drifting Volatility in Large
Bayesian VARs. *Journal of Business and Economic Statistics*, 34(3): 375-390.
[doi:10.1080/07350015.2015.1040116](https://doi.org/10.1080/07350015.2015.1040116)

Carriero, A., Clark, T. E. and Marcellino, M. (2019). Large Bayesian Vector Autoregressions with
Stochastic Volatility and Non-Conjugate Priors. *Journal of Econometrics*, 212(1): 137-154.
[doi:10.1016/j.jeconom.2019.04.024](https://doi.org/10.1016/j.jeconom.2019.04.024)

Carriero, A., Clark, T. E., Marcellino, M. and Mertens, E. (2024). Addressing COVID-19 Outliers
in BVARs with Stochastic Volatility. *Review of Economics and Statistics*, 106(5): 1403-1417.
[doi:10.1162/rest_a_01213](https://doi.org/10.1162/rest_a_01213)

Chan, J. C. C. (2023). Comparing Stochastic Volatility Specifications for Large Bayesian VARs.
*Journal of Econometrics*, 235(2): 1419-1446.
[doi:10.1016/j.jeconom.2022.11.003](https://doi.org/10.1016/j.jeconom.2022.11.003)

Chan, J. C. C. (forthcoming). *Bayesian Macroeconometrics: Methods and Applications*. Chapman &
Hall/CRC. [Code repository](https://github.com/joshuaccchan/bayesian-macroeconometrics)

Chan, J. C. C. and Eisenstat, E. (2015). Marginal Likelihood Estimation with the Cross-Entropy
Method. *Econometric Reviews*, 34(3): 256-285.
[doi:10.1080/07474938.2014.944474](https://doi.org/10.1080/07474938.2014.944474)

Cogley, T. and Sargent, T. J. (2005). Drifts and Volatilities: Monetary Policies and Outcomes in
the Post WWII US. *Review of Economic Dynamics*, 8(2): 262-302.
[doi:10.1016/j.red.2004.10.009](https://doi.org/10.1016/j.red.2004.10.009)

McCracken, M. W. and Ng, S. (2016). FRED-MD: A Monthly Database for Macroeconomic Research.
*Journal of Business and Economic Statistics*, 34(4): 574-589.
[doi:10.1080/07350015.2015.1086655](https://doi.org/10.1080/07350015.2015.1086655)

Stock, J. H. and Watson, M. W. (2016). Core Inflation and Trend Inflation. *Review of Economics
and Statistics*, 98(4): 770-784.
[doi:10.1162/REST_a_00608](https://doi.org/10.1162/REST_a_00608)

The BibTeX entry for Chan (2023) is in [`CITING.md`](../../CITING.md); those for the other papers
are:

```bibtex
@inproceedings{AR56,
  author    = {Anderson, T. W. and Rubin, H.},
  title     = {Statistical Inference in Factor Analysis},
  booktitle = {Proceedings of the Third {Berkeley} Symposium on Mathematical Statistics and Probability},
  volume    = {5},
  pages     = {111--150},
  year      = {1956},
  publisher = {University of California Press}
}

@article{CCM16,
  author  = {Carriero, A. and Clark, T. E. and Marcellino, M.},
  title   = {Common Drifting Volatility in Large {Bayesian} {VARs}},
  journal = {Journal of Business and Economic Statistics},
  year    = {2016},
  volume  = {34},
  number  = {3},
  pages   = {375--390},
  doi     = {10.1080/07350015.2015.1040116}
}

@article{CCM19,
  author  = {Carriero, A. and Clark, T. E. and Marcellino, M.},
  title   = {Large {Bayesian} Vector Autoregressions with Stochastic Volatility and Non-Conjugate Priors},
  journal = {Journal of Econometrics},
  year    = {2019},
  volume  = {212},
  number  = {1},
  pages   = {137--154},
  doi     = {10.1016/j.jeconom.2019.04.024}
}

@article{CCMM24,
  author  = {Carriero, A. and Clark, T. E. and Marcellino, M. and Mertens, E.},
  title   = {Addressing {COVID-19} Outliers in {BVARs} with Stochastic Volatility},
  journal = {Review of Economics and Statistics},
  year    = {2024},
  volume  = {106},
  number  = {5},
  pages   = {1403--1417},
  doi     = {10.1162/rest_a_01213}
}

@article{CE15,
  author  = {Chan, J. C. C. and Eisenstat, E.},
  title   = {Marginal Likelihood Estimation with the Cross-Entropy Method},
  journal = {Econometric Reviews},
  year    = {2015},
  volume  = {34},
  number  = {3},
  pages   = {256--285},
  doi     = {10.1080/07474938.2014.944474}
}

@article{CS05,
  author  = {Cogley, T. and Sargent, T. J.},
  title   = {Drifts and Volatilities: Monetary Policies and Outcomes in the Post {WWII} {US}},
  journal = {Review of Economic Dynamics},
  year    = {2005},
  volume  = {8},
  number  = {2},
  pages   = {262--302},
  doi     = {10.1016/j.red.2004.10.009}
}

@article{MN16,
  author  = {McCracken, M. W. and Ng, S.},
  title   = {{FRED-MD}: A Monthly Database for Macroeconomic Research},
  journal = {Journal of Business and Economic Statistics},
  year    = {2016},
  volume  = {34},
  number  = {4},
  pages   = {574--589},
  doi     = {10.1080/07350015.2015.1086655}
}

@article{SW16,
  author  = {Stock, J. H. and Watson, M. W.},
  title   = {Core Inflation and Trend Inflation},
  journal = {Review of Economics and Statistics},
  year    = {2016},
  volume  = {98},
  number  = {4},
  pages   = {770--784},
  doi     = {10.1162/REST_a_00608}
}
```
