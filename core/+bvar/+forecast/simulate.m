% bvar.forecast.simulate - h-step forecasts of a reduced-form VAR whose error
% covariance varies over time, from ONE posterior draw, with the log predictive
% likelihood of the outturn at each horizon. The counterpart of
% bvar.forecast.predictive, which covers the constant-covariance case exactly.
%
%   [yhat, lden, ljoint, sdev] = bvar.forecast.simulate(spec, draw, cfg)
%
%   spec : which covariance the errors have
%            'gauss' : constant, draw.Sig
%            'csv'   : exp(h_t)*Sig with h_t a zero-mean AR(1), the model of
%                      bvar.models.var_csv
%            'oisv'  : B0\diag(exp(h_t))/B0' with one zero-mean AR(1) per
%                      equation, the order-invariant model of bvar.models.var_sv
%   draw : one posterior draw. Every spec reads
%            A    : k x n coefficients, intercept first, k = n*p + 1
%          and its own volatility fields
%            'gauss' : Sig (n x n)
%            'csv'   : Sig (n x n), h_T, phi, sigh2 (scalars)
%            'oisv'  : impact (n x n, B0), h_T, phi, sig2 (1 x n each)
%   cfg  : ylag (n x p, most recent column first), H, and yobs (H x n, the
%          outturn) when the scores are wanted; rows of yobs that are NaN or
%          missing are skipped
%   yhat   : H x n conditional means given the simulated path
%   lden   : H x n log predictive likelihood of each variable, NaN where yobs
%            has no row
%   ljoint : H x 1 joint log predictive likelihood over all n variables
%   sdev   : H x n standard deviation of each variable at each horizon, given
%            the simulated volatility path. The predictive variance of a
%            variable follows from these and yhat by the law of total variance,
%            mean(sdev.^2) + var(yhat) over the draws
%
% ONE path is simulated per call, as the branches of bvar.forecast.iterate do:
% the volatility is advanced, the conditional mean and the density of the outturn
% are evaluated at the state reached, and a simulated observation then advances
% the state. The average of yhat over draws is the point forecast, and the log of
% the average of exp(lden) over draws is the predictive likelihood.
%
% See:
% Chan, J.C.C. (2023). Comparing Stochastic Volatility Specifications for Large
% Bayesian VARs, Journal of Econometrics, 235(2): 1419-1446.

function [yhat, lden, ljoint, sdev] = simulate(spec, draw, cfg)
spec = lower(char(string(spec)));
A = draw.A;
[k, n] = size(A);
p = (k - 1)/n;
assert(p == fix(p) && p >= 1, 'bvar:forecast:simulate:badDims', ...
    'draw.A must be (n*p+1) by n');
assert(isequal(size(cfg.ylag), [n p]), 'bvar:forecast:simulate:badLags', ...
    'cfg.ylag must be %d by %d', n, p);
H = cfg.H;
yobs = [];
if isfield(cfg, 'yobs'), yobs = cfg.yobs; end

switch spec
    case 'gauss'
        Sig = draw.Sig;
    case 'csv'
        Sig = draw.Sig;  h = draw.h_T;  phi = draw.phi;  sdh = sqrt(draw.sigh2);
        CSig0 = chol(Sig, 'lower');
    case 'oisv'
        B0 = draw.impact;  h = draw.h_T(:);  phi = draw.phi(:);  sdh = sqrt(draw.sig2(:));
    otherwise
        error('bvar:forecast:simulate:badSpec', ...
            'spec must be ''gauss'', ''csv'' or ''oisv''; got ''%s''', spec);
end

yhat = zeros(H, n);
lden = nan(H, n);
ljoint = nan(H, 1);
sdev = zeros(H, n);
x = [1, reshape(cfg.ylag, 1, [])];              % ylag is n x p, most recent first

for j = 1:H
        % the covariance at T+j, after advancing the volatility
    switch spec
        case 'gauss'
            CS = chol(Sig, 'lower');
        case 'csv'
            h = phi*h + sdh*randn;
            CS = exp(h/2)*CSig0;
        case 'oisv'
            h = phi.*h + sdh.*randn(n,1);
            CS = chol(B0\diag(exp(h))/B0', 'lower');
    end

    EY = x*A;
    yhat(j,:) = EY;
    dS = sum(CS.^2, 2)';                        % the diagonal of CS*CS'
    sdev(j,:) = sqrt(dS);
    if ~isempty(yobs) && size(yobs,1) >= j && all(isfinite(yobs(j,:)))
        u = yobs(j,:) - EY;
        lden(j,:) = -.5*log(2*pi*dS) - .5*u.^2./dS;
        z = CS\u';
        ljoint(j) = -n/2*log(2*pi) - sum(log(diag(CS))) - .5*(z'*z);
    end

        % one simulated observation advances the state
    Ysim = EY + (CS*randn(n,1))';
    x = [1, Ysim, x(2:end-n)];
end
end
