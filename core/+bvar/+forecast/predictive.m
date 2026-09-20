% bvar.forecast.predictive - mean and standard deviation of the h-step predictive
% distribution of a reduced-form VAR, one row per posterior draw. Given the
% coefficients and the error covariance matrix the h-step forecast is Gaussian:
% the mean iterates the VAR, and the variance is the sum of Psi_j*Sig*Psi_j' over
% j = 0 to h-1, with Psi_j the moving-average matrices.
%
%   [mu, sd] = bvar.forecast.predictive(A, Sig, ylag, H)
%   [mu, sd] = bvar.forecast.predictive(A, Sig, ylag, H, tgt)
%
%   A     : nsim x n*k draws of the reduced-form coefficients, each row the k x n
%           matrix stacked column by column with the intercept first, k = n*p + 1,
%           as bvar.structural.reduced_form returns them
%   Sig   : nsim x n x n draws of the error covariance matrix
%   ylag  : n x p matrix of the last p observations, most recent column first
%   H     : longest horizon
%   tgt   : variables to return (default 1:n)
%   mu,sd : nsim x numel(tgt) x H
%
% Each row conditions on one draw of the parameters, so the spread of mu over rows
% is parameter uncertainty and sd is the width of the conditional forecast density.
% The predictive density of a variable is the mixture over rows.
%
% See:
% Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and
% Applications, Chapman & Hall/CRC, Section 12.2.3.

function [mu, sd] = predictive(A, Sig, ylag, H, tgt)
nsim = size(A, 1);
n = size(Sig, 2);
k = size(A, 2)/n;
p = (k - 1)/n;
assert(p == fix(p) && p >= 1, 'bvar:forecast:predictive:badDims', ...
    'A must have n*(n*p+1) columns for an integer p');
assert(size(Sig, 1) == nsim && size(Sig, 3) == n, 'bvar:forecast:predictive:badSig', ...
    'Sig must be nsim x n x n');
assert(isequal(size(ylag), [n p]), 'bvar:forecast:predictive:badLags', ...
    'ylag must be %d by %d', n, p);
if nargin < 5 || isempty(tgt), tgt = 1:n; end
assert(all(ismember(tgt, 1:n)), 'bvar:forecast:predictive:badTarget', ...
    'tgt must index the %d variables', n);

nt = numel(tgt);
mu = zeros(nsim, nt, H);
sd = zeros(nsim, nt, H);
for d = 1:nsim
    Ad = reshape(A(d,:), k, n);
    Sd = reshape(Sig(d,:,:), n, n);
    c = Ad(1,:)';
    Phi = reshape(Ad(2:end,:)', n, n, p);
    Psi = zeros(n, n, H);  Psi(:,:,1) = eye(n);
    yl = ylag;  V = zeros(n);
    for h = 1:H
        yh = c;
        for l = 1:p, yh = yh + Phi(:,:,l)*yl(:,l); end
        yl = [yh, yl(:,1:p-1)];
        for l = 1:min(h-1, p), Psi(:,:,h) = Psi(:,:,h) + Phi(:,:,l)*Psi(:,:,h-l); end
        V = V + Psi(:,:,h)*Sd*Psi(:,:,h)';
        mu(d,:,h) = yh(tgt);
        sd(d,:,h) = sqrt(diag(V(tgt,tgt)));
    end
end
end
