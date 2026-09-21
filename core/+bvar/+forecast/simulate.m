% bvar.forecast.simulate - h-step predictive distribution of a reduced-form VAR
% with time-varying error covariance, from ONE posterior draw, with the log
% predictive likelihood of the outturn at each horizon.
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
%          outturn) when the scores are wanted; rows of yobs that are missing or
%          not finite are skipped
%   yhat   : H x n conditional mean, which does not depend on the volatility
%   lden   : H x n log predictive likelihood of each variable given the
%            simulated volatility path, NaN where yobs has no row
%   ljoint : H x 1 the same jointly over all n variables
%   sdev   : H x n predictive standard deviation given that path
%
% ONLY THE VOLATILITY IS SIMULATED. The data path is integrated out
% analytically, so lden and ljoint are exact given the simulated volatility
% path. Averaging exp(lden) over draws integrates over the parameters and the
% volatility together. With 'gauss' the result equals bvar.forecast.predictive,
% which covers the constant-covariance case, for the same draw to machine
% precision at every horizon.
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

    % ---- the covariance at each horizon, the only simulated part ----
Sg = cell(H, 1);
switch spec
    case 'gauss'
        for j = 1:H, Sg{j} = draw.Sig; end
    case 'csv'
        h = draw.h_T;  phi = draw.phi;  sdh = sqrt(draw.sigh2);
        for j = 1:H
            h = phi*h + sdh*randn;
            Sg{j} = exp(h)*draw.Sig;
        end
    case 'oisv'
        B0 = draw.impact;
        h = draw.h_T(:);  phi = draw.phi(:);  sdh = sqrt(draw.sig2(:));
        for j = 1:H
            h = phi.*h + sdh.*randn(n,1);
            Sg{j} = B0\diag(exp(h))/B0';
        end
    otherwise
        error('bvar:forecast:simulate:badSpec', ...
            'spec must be ''gauss'', ''csv'' or ''oisv''; got ''%s''', spec);
end

    % ---- the moving-average matrices, Psi(:,:,i+1) is Psi_i ----
cst = A(1,:)';
Phi = reshape(A(2:end,:)', n, n, p);
Psi = zeros(n, n, H);  Psi(:,:,1) = eye(n);
for j = 2:H
    for l = 1:min(j-1, p)
        Psi(:,:,j) = Psi(:,:,j) + Phi(:,:,l)*Psi(:,:,j-l);
    end
end

yhat = zeros(H, n);
lden = nan(H, n);
ljoint = nan(H, 1);
sdev = zeros(H, n);
yl = cfg.ylag;

for j = 1:H
        % the mean iterates the VAR and does not involve the volatility
    yh = cst;
    for l = 1:p, yh = yh + Phi(:,:,l)*yl(:,l); end
    yl = [yh, yl(:,1:p-1)];
    yhat(j,:) = yh';

        % the variance given the simulated path
    V = zeros(n);
    for i = 0:j-1
        V = V + Psi(:,:,i+1)*Sg{j-i}*Psi(:,:,i+1)';
    end
    V = (V + V')/2;
    dV = diag(V)';
    sdev(j,:) = sqrt(dV);

    if ~isempty(yobs) && size(yobs,1) >= j && all(isfinite(yobs(j,:)))
        u = yobs(j,:) - yh';
        lden(j,:) = -.5*log(2*pi*dV) - .5*u.^2./dV;
        CV = chol(V, 'lower');
        z = CV\u';
        ljoint(j) = -n/2*log(2*pi) - sum(log(diag(CV))) - .5*(z'*z);
    end
end
end
