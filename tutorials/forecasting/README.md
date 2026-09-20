# Does Modeling the Volatility Improve My Forecasts?

*Code: [`build.m`](build.m) and [`your_data.m`](your_data.m), which call
[`bvar.models.var_csv`](../../core/+bvar/+models/var_csv.m),
[`bvar.models.var_sv`](../../core/+bvar/+models/var_sv.m) and
[`bvar.forecast.simulate`](../../core/+bvar/+forecast/simulate.m). Method:
[Carriero, Clark and Marcellino (2016)](../../CITING.md#carriero-clark-and-marcellino-2016) and
[Chan, Koop and Yu (2024)](../../CITING.md#chan-koop-and-yu-2024).*

In this tutorial we forecast the same five quarterly US series as
[the stochastic volatility comparison](../sv_specification/), recursively and out of sample, from
three reduced-form BVARs: a homoskedastic one, one with a common volatility factor, and one with a
volatility process per equation. Over 140 forecasts from 1990, both volatility models beat the
homoskedastic benchmark on both criteria. The largest gains are concentrated in the pandemic
quarters, and they are not confined to them: grouping the forecasts by the quarter they are for,
about a quarter of the one-quarter-ahead gain and half of the four-quarter-ahead gain is earned
before 2020. The mechanism is visible in the forecast bands, where a homoskedastic VAR is too wide
in calm years and far too narrow in a crisis.

## Try It Now

Two commands, from the root of the repository:

| Command | What it produces | Time |
|---|---|---|
| `run tutorials/forecasting/your_data.m` | the same comparison over a short window with short chains | about a minute |
| `run tutorials/forecasting/build.m` | every number and figure on this page | 22 minutes |
| `run tutorials/forecasting/bench_density.m` | Table 1, the two ways of scoring a multi-step density | about a minute |

This workflow needs MATLAB with the Statistics and Machine Learning Toolbox.

![Cumulative log score difference](fig_cumscore.png)

*Figure 1: The running sum of the one-quarter-ahead joint log predictive likelihood of each
volatility model minus that of the homoskedastic VAR, by forecast origin. A rising line means the
model is forecasting better. Both climb steadily from 1990 to 2019, reaching about 200 and 260,
and then jump by roughly 500 in 2020.*

## The Three Models

All three are reduced-form VARs with four lags, differing only in the error covariance matrix:

- **Homoskedastic**, $`\mathbf{u}_t \sim N(\mathbf{0}, \mathbf{\Sigma})`$, under the natural
  conjugate prior.
- **VAR-CSV**, $`\mathbf{u}_t \sim N(\mathbf{0}, \mathrm{e}^{h_t}\mathbf{\Sigma})`$ with $`h_t`$ a
  zero-mean AR(1), the common volatility of Carriero, Clark and Marcellino (2016), under the same
  prior. One factor scales the whole matrix. The volatility path is drawn in one block by the
  accept-reject Metropolis-Hastings step of Chan (2020), which builds on the algorithm of
  Chan (2017).
- **VAR-OISV**, one log-volatility per equation with the order-invariant impact matrix of Chan,
  Koop and Yu (2024), under that paper's prior with the shrinkage estimated. The name distinguishes
  it from the Cholesky specification that [tutorial 3](../sv_specification/) calls VAR-SV.

None of the three depends on the order of the columns, so the comparison never asks the reader to
justify an ordering — which is what [the ordering tutorial](../variable_ordering/) is about. The
first two share a prior, so the difference between them is the volatility model alone; the third
brings its paper's prior with it, so it is a comparison of whole model-prior configurations, as in
the marginal likelihood comparison of [tutorial 3](../sv_specification/).

Each model is re-estimated at every origin on data up to that quarter, with 5,000 draws after
1,000 burn-in, and forecast one to four quarters ahead. Point forecasts are scored by RMSFE and
density forecasts by the log predictive likelihood, the log of the average of the predictive
density over the draws.

## How the Densities Are Computed

For the homoskedastic VAR the $`h`$-step predictive distribution is Gaussian in closed form, and
[`bvar.forecast.predictive`](../../core/+bvar/+forecast/predictive.m) returns its mean and variance
for each draw. When the covariance varies over time there is no closed form, because the
covariance at $`T+h`$ is not known at $`T`$. The natural thing is to simulate, and there are two
ways to do it.

The first simulates a whole path of the data for each draw and scores the outturn against it.
Both estimators are unbiased for the predictive *density*, and neither is unbiased for its
logarithm: by Jensen's inequality the log of a noisy density estimate sits below the log density,
the more so the noisier it is. The path-based estimator is the noisier one, because almost every
simulated path lands nowhere near an extreme outturn and the average is then carried by a few
lucky draws.

`bvar.forecast.simulate` does the second. Only the volatility path is simulated; given it, the
$`h`$-step distribution is still Gaussian, with the mean iterating the VAR and the variance

```math
\mathbf{V}_h = \sum_{i=0}^{h-1} \mathbf{\Psi}_i \mathbf{\Sigma}_{T+h-i} \mathbf{\Psi}_i',
```

so the density is evaluated exactly and only the volatility is integrated by Monte Carlo. What
comes out of the draws is still a mixture, not a Gaussian: conditioning on the coefficients and
the volatility path gives a normal, and averaging over parameter and volatility uncertainty does
not.

[`bench_density.m`](bench_density.m) runs the two estimators on identical posterior draws, at an
ordinary origin and at the pandemic one, over three simulation sizes and five seeds. It takes
about a minute.

*Table 1: Joint log score of the five variables from the same 5,000 posterior draws, averaged over
five simulation seeds, with the range over those seeds in parentheses. The posterior draws are
held fixed, so the range is simulation noise alone.*

| Origin | Horizon | Conditional Gaussian | Path based |
|---|---|---|---|
| 2015Q4, ordinary | 1 | −1.49 (0.04) | −1.49 (0.03) |
| 2015Q4, ordinary | 4 | −3.79 (0.04) | −3.76 (0.37) |
| 2019Q4, pandemic | 1 | −9.31 (0.08) | −9.31 (0.07) |
| 2019Q4, pandemic | 4 | −18.28 (0.32) | **−19.83 (1.30)** |

At one quarter ahead the two coincide, since no data path has been simulated yet. Beyond that the
path-based estimator is both noisier and biased downward, and both effects are worst where the
outturn is extreme. At 250 draws rather than 5,000 the pandemic four-quarter figure falls to
−28.98 with a range of 11.01 across seeds, against −19.27 and 1.81 for the conditional-Gaussian
estimator. A reader whose multi-step density scores jump around between seeds is most likely
looking at this.

As an implementation check the build also scores the homoskedastic model both ways, by simulation
and in closed form: they agree to `0.0000` at both horizons, over 140 and 137 forecasts and five
variables. That check confirms the code, not the Monte Carlo error that remains when the
volatility is uncertain, which is what the table above measures.

## What the Volatility Models Buy

Forecasts are grouped below by the quarter they are **for**, not by the quarter they were made in.
A four-quarter-ahead forecast made in 2019Q4 is a forecast of 2020Q4 and belongs with the pandemic;
grouping it by its origin would place four pandemic forecasts in the calm block and is the kind of
ambiguity that survives into a reader's own table. The scored counts differ by horizon, 140 at one
quarter and 137 at four, because the last three origins have no four-quarter outturn yet.

*Table 2: Accuracy relative to the homoskedastic VAR. The RMSFE column is the median percentage
gain across the five variables; the log score column is the mean gain in the joint log predictive
likelihood, per quarter.*

| Model | Horizon | Group | Forecasts | RMSFE gain | Log score gain |
|---|---|---|---|---|---|
| VAR-CSV | 1 | all | 140 | 4.9% | 5.76 |
| | 1 | targets through 2019 | 120 | 5.1% | 1.63 |
| | 1 | targets 2020 onwards | 20 | −0.9% | 30.50 |
| VAR-CSV | 4 | all | 137 | 5.0% | 2.68 |
| | 4 | targets through 2019 | 117 | 13.5% | 1.43 |
| | 4 | targets 2020 onwards | 20 | −0.0% | 9.98 |
| VAR-OISV | 1 | all | 140 | 8.5% | 5.62 |
| | 1 | targets through 2019 | 120 | 3.7% | 2.21 |
| | 1 | targets 2020 onwards | 20 | 22.6% | 26.11 |
| VAR-OISV | 4 | all | 137 | 17.6% | 2.07 |
| | 4 | targets through 2019 | 117 | 20.4% | 1.61 |
| | 4 | targets 2020 onwards | 20 | 8.7% | 4.81 |

Two readings follow, and they point in different directions.

**Density forecasts.** The gain is largest for pandemic targets, by a factor of twenty at one
quarter and of four to seven at four. It is not confined to them: multiplying each block's mean by
its count, the calm block accounts for 24% of VAR-CSV's total one-quarter gain and 34% of
VAR-OISV's, and at four quarters for 46% and 66%. The four-quarter gain is the smaller of the two
and the more evenly earned.

**Point forecasts.** The pattern is close to reversed. Most of the RMSFE gain is earned in calm
quarters — 13.5% and 20.4% at four quarters — while for pandemic targets VAR-CSV's point forecasts
are no better than the benchmark's at either horizon. Only VAR-OISV improves them, by 22.6% at one
quarter.

The two criteria therefore pick different models: VAR-CSV has the larger log score gain at both
horizons, VAR-OISV the larger RMSFE gain. These
are descriptive comparisons on 140 forecasts, and this page does not test whether the two
volatility models differ from each other.

## Why the Bands Move

![Predictive standard deviation of GDP growth](fig_psd.png)

*Figure 2: One-quarter-ahead predictive standard deviation of GDP growth at each origin. The
homoskedastic line is nearly flat; the two volatility models fall to about 2 in calm years and
rise to 27 and 18 in 2020.*

*Table 3: Median one-quarter-ahead predictive standard deviation of GDP growth, by the quarter forecast.*

| Model | Targets through 2019 | Targets 2020 onwards |
|---|---|---|
| Homoskedastic | 2.80 | 3.93 |
| VAR-CSV | 2.05 | 3.27 |
| VAR-OISV | 2.36 | 3.05 |

The homoskedastic VAR carries the widest interval in both blocks. It fits one covariance matrix
to thirty years, so in calm quarters its density is too diffuse and
loses log points steadily, and in 2020 it is far too tight and loses them in bulk. Its band does
widen after 2020, from 2.80 to 3.93, but only because the pandemic observations have entered the
sample and stay in it: the widening is permanent, and it arrives a quarter late.

## Checking the Estimates

The two chains are re-run at every origin, so mixing matters. At the last origin, with 5,000 draws
and a bandwidth of 200, the inefficiency factors are 8, 23 and 62 for VAR-CSV's persistence, its
volatility variance and $`\Sigma_{11}`$, and 21, 62 and 12 for VAR-OISV's first persistence, its
first volatility variance and the log-volatility at the end of the sample. A factor of 62 means
5,000 draws carry the information of about 80 independent ones, which is thin for a single
posterior summary and adequate for an average over 140 origins. Section 6.5 of *Bayesian
Macroeconometrics* sets out the diagnostic, and [ex13](../../examples/ex13_mcmc_diagnostics.m)
applies it in detail.

## Applying the Method to Your Data

The script [`your_data.m`](your_data.m) runs the same comparison on any dataset. Set the file, the
columns, the lag length, the evaluation window and the chain lengths in its settings block. The
loop is three calls per origin:

```matlab
res = bvar.models.var_csv(Y0, Y, p, 'nsim', nsim, 'burnin', burnin, 'draws', true);
dr = struct('A', reshape(res.draws.A(d,:), k, n), 'Sig', reshape(res.draws.Sig(d,:), n, n), ...
            'h_T', res.draws.h(d,end), 'phi', res.draws.phi(d), 'sigh2', res.draws.sigh2(d));
[yhat, lden, ljoint] = bvar.forecast.simulate('csv', dr, cfg);
```

with `cfg` holding the last `p` observations, the longest horizon and the outturn. `bvar.models.var_sv`
returns the same shape of draws for the order-invariant model, with `impact`, `h_T`, `phi` and
`sig2` in place of the common factor's fields, and `simulate` takes `'oisv'` for it. The script
ends by writing the scores to `outdir`, which defaults to `tempdir`.

A recursive exercise costs one estimation per origin per model, so its runtime is the product of
four numbers: origins, models, sweeps and the cost of a sweep. Halving the origins or the sweeps
halves the wait, and the scores move very little.

## Reproducing the Results

To reproduce all results on this page, run the build script from the root of the repository:

```matlab
run tutorials/forecasting/build.m
```

The computation took 22.6 minutes using MATLAB R2025b on a computer with an Intel Core Ultra 7
255U processor and 32 GB of RAM. All results are printed in [`build_log.txt`](build_log.txt) or
computed from numbers printed there, the figures are saved in the same folder, and the scores of
every individual forecast, with the quarter it was made in and the quarter it is for, are in
`scores_by_origin.mat`, so a regrouping or a question about one quarter needs no rerun.
[`bench_density.m`](bench_density.m) produces Table 1 and is not part of the build.

## References

Carriero, A., Clark, T. E. and Marcellino, M. (2016). Common drifting volatility in large Bayesian
VARs. *Journal of Business and Economic Statistics*, 34(3): 375-390.

Chan, J. C. C. (2017). The stochastic volatility in mean model with time-varying parameters: an
application to inflation modeling. *Journal of Business and Economic Statistics*, 35(1): 17-28.

Chan, J. C. C. (2020). Large Bayesian VARs: a flexible Kronecker error covariance structure.
*Journal of Business and Economic Statistics*, 38(1): 68-79.

Chan, J. C. C. (2023). Comparing stochastic volatility specifications for large Bayesian VARs.
*Journal of Econometrics*, 235(2): 1419-1446.

Chan, J. C. C., Koop, G. and Yu, X. (2024). Large order-invariant Bayesian VARs with stochastic
volatility. *Journal of Business and Economic Statistics*, 42(2): 825-837.
