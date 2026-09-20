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
volatility process per equation. Over 140 origins from 1990, both volatility models beat the
homoskedastic benchmark on both criteria, and the answer separates by horizon. Four quarters
ahead the density gain is modest, about two log points a quarter, but it accumulates steadily and
a Diebold-Mariano test rejects equal accuracy. One quarter ahead the average gain is three times
larger, and yet the same test does not reject, because almost all of it arrives in two quarters of
2020. The mechanism is visible in the forecast bands: a homoskedastic VAR is too wide in calm
years and far too narrow in a crisis.

## Try It Now

Two commands, from the root of the repository:

| Command | What it produces | Time |
|---|---|---|
| `run tutorials/forecasting/your_data.m` | the same comparison over a short window with short chains | about a minute |
| `run tutorials/forecasting/build.m` | every number and figure on this page | 22 minutes |

This workflow needs MATLAB with the Statistics and Machine Learning Toolbox.

![Cumulative log score difference](fig_cumscore.png)

*Figure 1: The running sum of the one-quarter-ahead joint log predictive likelihood of each
volatility model minus that of the homoskedastic VAR. A rising line means the model is forecasting
better. Both rise steadily from 1990 to 2019 and then jump by about 500 log points in 2020.*

## The Three Models

All three are reduced-form VARs with four lags, differing only in the error covariance matrix:

- **Homoskedastic**, $`\mathbf{u}_t \sim N(\mathbf{0}, \mathbf{\Sigma})`$, under the natural
  conjugate prior.
- **VAR-CSV**, $`\mathbf{u}_t \sim N(\mathbf{0}, \mathrm{e}^{h_t}\mathbf{\Sigma})`$ with $`h_t`$ a
  zero-mean AR(1), the common volatility of Carriero, Clark and Marcellino (2016), under the same
  prior. One factor scales the whole matrix.
- **VAR-SV**, one log-volatility per equation with the order-invariant impact matrix of Chan, Koop
  and Yu (2024), under that paper's prior with the shrinkage estimated.

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

The first simulates a whole path of the data for each draw and scores the outturn against it. It
is unbiased, but its variance grows with the horizon and collapses in the tails: almost every
simulated path lands nowhere near an extreme outturn, so the average is carried by one lucky draw.
On this data that estimator agrees with the closed form to machine precision at one quarter ahead
and is wrong by up to **309 log points** at four quarters ahead in the pandemic quarters.

`bvar.forecast.simulate` does the second. Only the volatility path is simulated; given it, the
$`h`$-step distribution is still Gaussian, with the mean iterating the VAR and the variance

```math
\mathbf{V}_h = \sum_{i=0}^{h-1} \mathbf{\Psi}_i \mathbf{\Sigma}_{T+h-i} \mathbf{\Psi}_i',
```

so the density is evaluated exactly and only the volatility is integrated by Monte Carlo. The
build checks this by scoring the homoskedastic model both ways: the simulated and the closed-form
scores agree to `0.0000` at both horizons, over 140 origins and five variables.

## What the Volatility Models Buy

*Table 1: Accuracy relative to the homoskedastic VAR. The RMSFE column is the median percentage
gain across the five variables; the log score column is the mean gain in the joint log predictive
likelihood, per quarter.*

| Model | Horizon | RMSFE gain, percent | Log score gain |
|---|---|---|---|
| VAR-CSV | 1 | 4.9 | 5.76 |
| VAR-CSV | 4 | 5.0 | 2.68 |
| VAR-SV | 1 | 8.5 | 5.62 |
| VAR-SV | 4 | 17.6 | 2.07 |

*Table 2: The same, split at 2020. The first block has 121 origins and the second 19.*

| Model | Horizon | RMSFE gain through 2019 | Log score gain through 2019 | RMSFE gain 2020 on | Log score gain 2020 on |
|---|---|---|---|---|---|
| VAR-CSV | 1 | 4.0 | 1.61 | −0.6 | **32.18** |
| VAR-CSV | 4 | 12.5 | 2.73 | 7.1 | 2.31 |
| VAR-SV | 1 | 3.4 | 2.17 | 22.7 | **27.61** |
| VAR-SV | 4 | 19.8 | 2.09 | 12.3 | 1.94 |

The one-quarter-ahead density gain is 1.6 to 2.2 log points a quarter through 2019 and about
thirty from 2020 on. Four quarters ahead it is close to two in both blocks: the gain is smaller,
but it is there in every period rather than in a few.

*Table 3: Diebold-Mariano tests against the homoskedastic VAR on the joint log score, with a
Newey-West long-run variance at $`h-1`$ lags. A statistic beyond 1.96 rejects equal accuracy at
the 5% level.*

| Model | Horizon | Mean gain | DM |
|---|---|---|---|
| VAR-CSV | 1 | 5.76 | 1.42 |
| VAR-CSV | 4 | 2.68 | **2.40** |
| VAR-SV | 1 | 5.62 | 1.74 |
| VAR-SV | 4 | 2.07 | **3.88** |

The tests reject at four quarters and not at one, which is the opposite of what the mean gains
alone would suggest. A Diebold-Mariano test asks whether the advantage is consistent quarter by
quarter, and at one quarter ahead it is not: Figure 1 shows a steady climb of about 200 log points
over thirty years and then a vertical jump in 2020. A mean of 5.76 built that way has a large
standard error. Reporting the average without the test, or the test without the figure, would both
mislead.

## Why the Bands Move

![Predictive standard deviation of GDP growth](fig_psd.png)

*Figure 2: One-quarter-ahead predictive standard deviation of GDP growth at each origin. The
homoskedastic line is nearly flat; the two volatility models fall to about 2 in calm years and
rise to 27 and 18 in 2020.*

*Table 4: Median one-quarter-ahead predictive standard deviation of GDP growth.*

| Model | Through 2019 | 2020 onwards |
|---|---|---|
| Homoskedastic | 2.80 | 3.93 |
| VAR-CSV | 2.04 | 3.29 |
| VAR-SV | 2.36 | 3.14 |

The homoskedastic VAR carries the widest interval in both blocks. It fits one covariance matrix
to thirty years, so in calm quarters its density is too diffuse and
loses log points steadily, and in 2020 it is far too tight and loses them in bulk. Its band does
widen after 2020, from 2.80 to 3.93, but only because the pandemic observations have entered the
sample and stay in it: the widening is permanent, and it arrives a quarter late.

## Checking the Estimates

The two chains are re-run at every origin, so mixing matters. At the last origin, with 5,000 draws
and a bandwidth of 200, the inefficiency factors are 8, 23 and 62 for VAR-CSV's persistence, its
volatility variance and $`\Sigma_{11}`$, and 21, 62 and 12 for VAR-SV's first persistence, its
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

The computation took 22.4 minutes using MATLAB R2025b on a computer with an Intel Core Ultra 7
255U processor and 32 GB of RAM. All results are printed in [`build_log.txt`](build_log.txt) or
computed from numbers printed there, and the figures are saved in the same folder.

## References

Carriero, A., Clark, T. E. and Marcellino, M. (2016). Common drifting volatility in large Bayesian
VARs. *Journal of Business and Economic Statistics*, 34(3): 375-390.

Chan, J. C. C. (2023). Comparing stochastic volatility specifications for large Bayesian VARs.
*Journal of Econometrics*, 235(2): 1419-1446.

Chan, J. C. C., Koop, G. and Yu, X. (2024). Large order-invariant Bayesian VARs with stochastic
volatility. *Journal of Business and Economic Statistics*, 42(2): 825-837.
