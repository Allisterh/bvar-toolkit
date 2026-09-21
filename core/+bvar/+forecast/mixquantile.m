% bvar.forecast.mixquantile - quantiles of an equally weighted mixture of normals,
% which is what a Bayesian predictive distribution is once the draws are in hand:
% each draw contributes a normal, and the predictive density is their average.
%
%   q = bvar.forecast.mixquantile(mu, sd, p)
%
%   mu, sd : one component per draw, the same length. bvar.forecast.simulate and
%            bvar.forecast.predictive return exactly this pair for a variable at
%            a horizon
%   p      : probabilities in (0,1), any shape
%   q      : the quantiles, the same shape as p
%
% The mixture cdf, mean(normcdf((x - mu)./sd)), is continuous and increasing, so
% the quantile is found by bisection from a bracket that covers every component.
% Averaging the component quantiles instead would be wrong, and so would fitting
% one normal to the mixture's mean and variance: the mixture is skewed and
% fat-tailed whenever the draws disagree, which is the case worth reporting.
%
% For the probability of an event rather than a quantile, the cdf is one line:
% mean(normcdf((x - mu)./sd)) is the predictive probability of falling below x.

function q = mixquantile(mu, sd, p)
mu = mu(:);  sd = sd(:);
assert(numel(mu) == numel(sd), 'bvar:forecast:mixquantile:badPair', ...
    'mu and sd must have the same length');
assert(all(sd > 0), 'bvar:forecast:mixquantile:badSd', ...
    'every component standard deviation must be positive');
assert(all(p(:) > 0 & p(:) < 1), 'bvar:forecast:mixquantile:badProb', ...
    'probabilities must lie strictly between 0 and 1');

cdf = @(x) mean(normcdf((x - mu)./sd));
lo = min(mu - 12*sd);
hi = max(mu + 12*sd);
q = zeros(size(p));
for i = 1:numel(p)
    a = lo;  b = hi;
    for it = 1:200                              % bisection to machine tolerance
        m = (a + b)/2;
        if cdf(m) < p(i), a = m; else, b = m; end
        if b - a < 1e-12*max(1, abs(m)), break; end
    end
    q(i) = (a + b)/2;
end
end
