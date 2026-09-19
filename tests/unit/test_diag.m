function test_diag
% bvar.diag.specvar0, inefficiency_factor, mcse and geweke must equal, bitwise,
% the book's chapter 6 functions (geweke is geweke_diag there), held verbatim at
% the end of this file from bayesian-macroeconometrics commit 92bcf76, on chains
% with several columns and truncation lags. They must also do what their
% definitions say: an inefficiency factor of 1 at L = 0 and close to its windowed
% value for an AR(1) chain, MCSE^2 = IF*var/R, a Geweke Z that rejects about 5%
% of independent chains at the 5% level and detects a shift in the mean.

rng(20260919, 'twister');
R = 3000;
phis = [0 .5 .9 .99];
X = zeros(R, numel(phis) + 1);
for j = 1:numel(phis)
    X(:,j) = filter(1, [1 -phis(j)], randn(R,1));
end
X(:,end) = cumsum(randn(R,1))/10 + randn(R,1);      % a column with no stationary mean

    % bitwise against the book's code
for L = [0 1 7 50]
    assert(isequal(bvar.diag.inefficiency_factor(X, L), inefficiency_factor(X, L)), ...
        'inefficiency_factor differs from the book at L = %d', L);
    assert(isequal(bvar.diag.mcse(X, L), mcse(X, L)), 'mcse differs from the book at L = %d', L);
    for j = 1:size(X,2)
        assert(isequal(bvar.diag.specvar0(X(:,j), L), specvar0(X(:,j), L)), ...
            'specvar0 differs from the book at L = %d, column %d', L, j);
    end
end
args = {{}, {.2, .6}, {[], [], 10}, {.1, .5, 'auto'}, {.3, .7, 0}};
for ia = 1:numel(args)
    [Z1, p1, i1] = bvar.diag.geweke(X, args{ia}{:});
    [Z2, p2, i2] = geweke_diag(X, args{ia}{:});
    assert(isequal(Z1, Z2) && isequal(p1, p2) && isequal(i1, i2), ...
        'geweke differs from the book''s geweke_diag, argument set %d', ia);
end

    % a matrix is one series per column, a vector of either orientation one chain
S = bvar.diag.specvar0(X, 7);
for j = 1:size(X,2)
    assert(isequal(S(j), specvar0(X(:,j), 7)), 'specvar0: column %d of a matrix', j);
end
x = X(:,3);
assert(isequal(bvar.diag.specvar0(x', 7), specvar0(x, 7)) ...
    && isequal(bvar.diag.inefficiency_factor(x', 50), inefficiency_factor(x, 50)) ...
    && isequal(bvar.diag.mcse(x', 50), mcse(x, 50)) ...
    && isequal(bvar.diag.geweke(x'), geweke_diag(x)), 'a row vector must be one chain');

    % properties
assert(max(abs(bvar.diag.inefficiency_factor(X, 0) - 1)) < 1e-12, ...
    'at L = 0 the inefficiency factor is 1');
IF = bvar.diag.inefficiency_factor(X, 50);
M = bvar.diag.mcse(X, 50);
assert(max(abs(M.^2*R./(IF.*var(X, 1)) - 1)) < 1e-12, 'MCSE^2 must equal IF*var/R');

    % AR(1) with coefficient .9 has inefficiency factor 19; the Bartlett window with
    % L = 200 estimates 1 + 2*sum((1 - l/201)*.9^l), about 18.1. The estimate's
    % standard deviation is about 3.7% of it with 200,000 draws.
rng(1, 'twister');
x = filter(1, [1 -.9], randn(2e5, 1));
L = 200; ell = 1:L;
target = 1 + 2*sum((1 - ell/(L+1)).*.9.^ell);
IFx = bvar.diag.inefficiency_factor(x, L);
assert(abs(IFx/target - 1) < .15, 'AR(1): inefficiency factor %.2f, windowed value %.2f', IFx, target);

    % Z on 2,000 chains of 1,000 independent draws, and on a mean shift of half a
    % standard deviation halfway through a chain
rng(2, 'twister');
rate = mean(abs(bvar.diag.geweke(randn(1000, 2000))) > 1.96);
assert(rate > .03 && rate < .08, 'geweke: rejection rate %.3f on independent draws', rate);
y = randn(4000, 1);
y(2001:end) = y(2001:end) + .5;
assert(abs(bvar.diag.geweke(y)) > 5, 'geweke: a mean shift of half a standard deviation was missed');

    % input checks
expect_error(@() bvar.diag.inefficiency_factor(X, -1), 'bvar:diag:specvar0:badLag');
expect_error(@() bvar.diag.mcse(X, 2.5), 'bvar:diag:specvar0:badLag');
expect_error(@() bvar.diag.specvar0(X, R), 'bvar:diag:specvar0:badLag');
expect_error(@() bvar.diag.geweke(X, .5, .4), 'bvar:diag:geweke:badSegments');
expect_error(@() bvar.diag.geweke(X, [], [], 'nw'), 'bvar:diag:geweke:badLag');
expect_error(@() bvar.diag.geweke(X(1:15,:)), 'bvar:diag:geweke:shortChain');
end

function expect_error(f, id)
try
    f();
catch err
    assert(strcmp(err.identifier, id), 'expected %s, got %s: %s', id, err.identifier, err.message);
    return
end
error('expected error %s, none thrown', id);
end

% -------------------------------------------------------------------------
% The book's functions, verbatim from code/matlab/chapter06/ of
% bayesian-macroeconometrics at commit 92bcf76.
% -------------------------------------------------------------------------

function S = specvar0(x, L)
% specvar0.m
% Estimates the long-run variance of the sample mean of x using a
% Bartlett-window spectral-variance estimator at frequency zero:
%       S = (gamma_0 + 2 * sum_{ell=1}^{L} w_ell * gamma_ell) / T,
% where gamma_ell is the lag-ell sample autocovariance and the weights
% w_ell = 1 - ell/(L+1) are Bartlett (Newey-West) weights.
%
% Inputs:
%   x : T-by-1 vector (demeaned internally)
%   L : truncation lag (nonnegative integer)
%
% Output:
%   S : long-run variance of the sample mean of x
x = x(:);
T = length(x);
x = x - mean(x);

% autocovariances up to lag L
gamma0 = (x' * x) / T;
Sx = gamma0;

for ell = 1:L
    w = 1 - ell/(L+1);  % Bartlett weight
    gamma = (x(1+ell:end)' * x(1:end-ell)) / T;
    Sx = Sx + 2 * w * gamma;
end

% long-run variance of the mean
S = Sx / T;
end

function IF = inefficiency_factor(draws, L)
% inefficiency_factor.m
% Computes inefficiency factors (integrated autocorrelation times) for
% MCMC output. For each column of draws, IF(j) = Omega_j / sigma2_j,
% where sigma2_j is the marginal variance and Omega_j is the long-run
% variance estimated by the spectral variance at zero (Bartlett window).
% Requires specvar0.m.
%
% Inputs:
%   draws : R-by-k matrix of MCMC draws (each column is a scalar sequence)
%   L     : truncation lag for the spectral variance estimator
%
% Output:
%   IF    : 1-by-k vector of inefficiency factors
[R, k] = size(draws);
IF = zeros(1, k);
for j = 1:k
    x = draws(:, j);
    sigma2 = var(x, 1);          % marginal variance (population version)
    Omega  = specvar0(x, L) * R; % long-run variance of x^{(r)}
    IF(j) = Omega / sigma2;
end
end

function MCSE = mcse(draws, L)
% mcse.m
% Computes Monte Carlo standard errors (MCSEs) for posterior means
% obtained from MCMC output. For each column of draws, MCSE(j) is
% sqrt(Omega_j / R), where Omega_j is the long-run variance estimated
% by the spectral variance at zero. Requires specvar0.m.
%
% Inputs:
%   draws : R-by-k matrix of MCMC draws
%   L     : truncation lag for the spectral variance estimator
%
% Output:
%   MCSE  : 1-by-k vector of Monte Carlo standard errors
[R, k] = size(draws);
MCSE = zeros(1, k);
for j = 1:k
    x = draws(:, j);
    Omega = specvar0(x, L) * R; % long-run variance of x^{(r)}
    MCSE(j) = sqrt(Omega / R);
end
end

function [Z, pval, info] = geweke_diag(draws, a, b, Lrule)
% geweke_diag.m
% Computes Geweke's convergence diagnostic for MCMC output. For each
% column of draws, the chain is split into an early segment A (first
% fraction a) and a late segment B (last fraction 1-b), the segment
% means are compared, and the Z-statistic
%       Z = (mean_A - mean_B) / sqrt(S_A + S_B)
% is computed, where S_A and S_B are spectral-variance estimates of the
% long-run variances of the two segment means. Under stationarity, Z is
% approximately standard normal; large |Z| provides evidence against
% convergence for the chosen summary. Requires specvar0.m.
%
% Inputs:
%   draws : R-by-k matrix of posterior draws (each column is a scalar
%           summary computed from MCMC output)
%   a     : fraction for early segment (default 0.10)
%   b     : fraction defining late segment, B = floor(b*R):R
%           (default 0.50)
%   Lrule : truncation-lag rule for the spectral variance estimator,
%           either 'auto' (default) or a nonnegative integer L
%
% Outputs:
%   Z     : 1-by-k vector of Geweke Z-statistics
%   pval  : 1-by-k vector of two-sided p-values (normal approximation)
%   info  : struct with fields A, B, LA, LB, a, b giving the segment
%           indices and chosen truncation lags
if nargin < 2 || isempty(a), a = 0.10; end
if nargin < 3 || isempty(b), b = 0.50; end
if nargin < 4 || isempty(Lrule), Lrule = 'auto'; end

[R, k] = size(draws);
if ~(0 < a && a < b && b < 1)
    error('Require 0 < a < b < 1.');
end

A = 1:floor(a*R);
B = floor(b*R):R;

nA = length(A);
nB = length(B);

% Choose truncation lags
if ischar(Lrule) || isstring(Lrule)
    if strcmpi(Lrule,'auto')
        LA = max(0, floor(4*(nA/100)^(2/9)));
        LB = max(0, floor(4*(nB/100)^(2/9)));
    else
        error('Unknown Lrule. Use ''auto'' or an integer.');
    end
else
    LA = max(0, floor(Lrule));
    LB = LA;
end

Z = nan(1,k);
pval = nan(1,k);

for j = 1:k
    xA = draws(A,j);
    xB = draws(B,j);

    mA = mean(xA);
    mB = mean(xB);

    SA = specvar0(xA, LA);  % long-run variance of mean for segment A
    SB = specvar0(xB, LB);  % long-run variance of mean for segment B

    Z(j) = (mA - mB) / sqrt(SA + SB);

    % Two-sided p-value under N(0,1)
    pval(j) = 2 * (1 - normcdf(abs(Z(j))));
end

info.A = A;
info.B = B;
info.LA = LA;
info.LB = LB;
info.a = a;
info.b = b;
end
