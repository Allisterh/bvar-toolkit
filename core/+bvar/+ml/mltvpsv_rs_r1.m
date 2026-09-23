% bvar.ml.mltvpsv_rs_r1 - log marginal likelihood of the RS-VAR-R1 model of Chan and
% Eisenstat (2018) by importance sampling. The importance density of each variance is
% inverse-gamma, that of the coefficients Gaussian and that of each row of the
% transition matrix Dirichlet, fitted to the posterior draws; the regimes are integrated
% out by bvar.ml.intlike_rsvar. The M importance weights are split into 20 batches: lml
% is the mean of the 20 batch estimates of the log marginal likelihood, and lmlstd is
% their standard error.
%
%   [lml, lmlstd, out] = bvar.ml.mltvpsv_rs_r1(Y, store_theta, store_Sig, store_P, ...
%       prior, bigX, M, bugcompat)
%
%   Y     : T*n x 1 data, stacked by period
%   store_*: posterior draws, one row per draw
%   prior : log prior density, prior(theta, Sig, P)
%   M     : importance draws, rounded up to a multiple of 20
%   out   : store_w, the M log weights; bigml, the 20 batch estimates
%   bugcompat : false (default) starts the Hamilton filter from p(s_1 = j) = 1/r;
%           true reproduces the published routine, which starts from 1/3 whatever r
%           is and so shifts every log likelihood, and the log marginal likelihood,
%           by log(r/3).
%
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for
% time-varying parameter VARs with stochastic volatility, Journal of Applied
% Econometrics, 33(4), 509-532.

function [ml, mlstd, out] = mltvpsv_rs_r1(Y,store_theta,store_Sig,store_P,prior,bigX,M,bugcompat)
if nargin < 8
    bugcompat = false;
end
M = 20*ceil(M/20);
r = size(store_P,2);
n = size(store_Sig,2)/r;
k = size(store_theta,2);
T = length(Y)/n;

shortY = reshape(Y,n,T)';
tmp = zeros(n*r,2);
for i=1:n*r
    tmp(i,:) = gamfit(1./store_Sig(:,i));
end
nuhat = tmp(:,1); Shat = 1./tmp(:,2);
thetahat = mean(store_theta)';
thetastd = chol(cov(store_theta),'lower');
thetapre = thetastd'\(thetastd\speye(k));
alphat = zeros(r,r);
for i=1:r
    alphat(i,:) = bvar.ml.dirifit(squeeze(store_P(:,i,:)))';
end

big_Sig = zeros(M,n*r);
for i=1:n*r
    big_Sig(:,i) = 1./gamrnd(nuhat(i),1./Shat(i),M,1);
end
big_theta = repmat(thetahat',M,1) + (thetastd*randn(k,M))';
big_P = zeros(M,r,r);
for i=1:r
    big_P(:,i,:) = bvar.util.dirirnd(alphat(i,:)',M);
end

store_w = zeros(M,1);
if bugcompat
    p1 = 1/3;
else
    p1 = 1/r;
end

cIS = -.5*k*log(2*pi) - sum(log(diag(thetastd))) + nuhat'*log(Shat) - sum(gammaln(nuhat));
gIS = @(the,s)+ cIS -.5*(the-thetahat)'*thetapre*(the-thetahat) ...
    -(nuhat+1)'*log(s) - sum(Shat./s);

for isim = 1:M
    Sig = big_Sig(isim,:)';
    theta = big_theta(isim,:)';
    P = squeeze(big_P(isim,:,:));
    g_IS_P = 0;
    for i=1:r
        g_IS_P = g_IS_P + bvar.ml.ldiripdf(P(i,:),alphat(i,:));
    end
    llike = bvar.ml.intlike_rsvar(shortY,bigX,repmat(theta,1,r),reshape(Sig,n,r),P,p1);
    store_w(isim) = llike + prior(theta,Sig,P) - (gIS(theta,Sig) + g_IS_P);
end
shortw = reshape(store_w,M/20,20);
maxw = max(shortw);

bigml = log(mean(exp(shortw-repmat(maxw,M/20,1)),1)) + maxw;
ml = mean(bigml);
mlstd = std(bigml)/sqrt(20);
out = struct('store_w',store_w, 'bigml',bigml);
end
