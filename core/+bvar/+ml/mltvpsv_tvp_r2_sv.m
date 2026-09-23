% bvar.ml.mltvpsv_tvp_r2_sv - log marginal likelihood of the TVP-R2-SV model of Chan and
% Eisenstat (2018) by importance sampling. The importance density of each variance is
% inverse-gamma and that of gam and of each initial state Gaussian, fitted to the
% posterior draws; the coefficient states beta_t and the log-volatilities are integrated
% out by bvar.ml.intlike_tvpsv, applied to Y - W*gam with Xtilde as the regressors. The
% M importance weights are split into 20 batches: lml is the mean of the 20 batch
% estimates of the log marginal likelihood, and lmlstd is their standard error.
%
%   [lml, lmlstd, out] = bvar.ml.mltvpsv_tvp_r2_sv(Y, store_gam, store_Sigbeta, ...
%       store_Sigh, store_h0, store_beta0, prior, Xtilde, W, M)
%
%   Y     : T*n x 1 data, stacked by period
%   store_*: posterior draws, one row per draw
%   prior : log prior density, prior(gam, Sigbeta, Sigh, beta0, h0)
%   M     : importance draws, rounded up to a multiple of 20
%   out   : store_w, the M log weights; bigml, the 20 batch estimates
%
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for
% time-varying parameter VARs with stochastic volatility, Journal of Applied
% Econometrics, 33(4), 509-532.

function [ml, mlstd, out] = mltvpsv_tvp_r2_sv(Y,store_gam,store_Sigbeta,store_Sigh,...
    store_h0,store_beta0,prior,Xtilde,W,M)
M = 20*ceil(M/20);
n = size(store_Sigh,2);
kbeta = size(store_Sigbeta,2);
kgam = size(store_gam,2);

temp = zeros(kbeta,2);
for i=1:kbeta
    temp(i,:) = gamfit(1./store_Sigbeta(:,i));
end
nubetahat = temp(:,1); Sbetahat = 1./temp(:,2);
temp = zeros(n,2);
for i=1:n
    temp(i,:) = gamfit(1./store_Sigh(:,i));
end
nuhhat = temp(:,1); Shhat = 1./temp(:,2);

gamhat = mean(store_gam)';
gamstd = chol(cov(store_gam),'lower');
gampre = gamstd'\(gamstd\speye(kgam));
beta0hat = mean(store_beta0)';
beta0std = chol(cov(store_beta0),'lower');
beta0pre = beta0std'\(beta0std\speye(kbeta));
h0hat = mean(store_h0)';
h0std = chol(cov(store_h0),'lower');
h0pre = h0std'\(h0std\speye(n));

big_Sigbeta = zeros(M,kbeta);
for i=1:kbeta
    big_Sigbeta(:,i) = 1./gamrnd(nubetahat(i),1./Sbetahat(i),M,1);
end
big_Sigh = zeros(M,n);
for i=1:n
    big_Sigh(:,i) = 1./gamrnd(nuhhat(i),1./Shhat(i),M,1);
end
big_gam = repmat(gamhat',M,1) + (gamstd*randn(kgam,M))';
big_beta0 = repmat(beta0hat',M,1) + (beta0std*randn(kbeta,M))';
big_h0 = repmat(h0hat',M,1) + (h0std*randn(n,M))';

store_w = zeros(M,1);
cIS = -.5*kgam*log(2*pi) - sum(log(diag(gamstd))) ...
    + nubetahat'*log(Sbetahat) - sum(gammaln(nubetahat)) ...
    + nuhhat'*log(Shhat) - sum(gammaln(nuhhat)) ...
    -.5*(kbeta+n)*log(2*pi) - sum(log(diag(beta0std))) - sum(log(diag(h0std)));
gIS = @(g,sb,sh,a0,b0) cIS -.5*(g-gamhat)'*gampre*(g-gamhat) ...
    -(nubetahat+1)'*log(sb) - sum(Sbetahat./sb) -(nuhhat+1)'*log(sh) - sum(Shhat./sh) ...
    -.5*(a0-beta0hat)'*beta0pre*(a0-beta0hat) -.5*(b0-h0hat)'*h0pre*(b0-h0hat);

for loop = 1:M
    gam = big_gam(loop,:)';
    Sigbeta = big_Sigbeta(loop,:)';
    Sigh = big_Sigh(loop,:)';
    beta0 = big_beta0(loop,:)';
    h0 = big_h0(loop,:)';

    llike = bvar.ml.intlike_tvpsv(Y-W*gam,Sigbeta,Sigh,Xtilde,h0,beta0);
    store_w(loop) = llike + prior(gam,Sigbeta,Sigh,beta0,h0) ...
        - gIS(gam,Sigbeta,Sigh,beta0,h0);
end
shortw = reshape(store_w,M/20,20);
maxw = max(shortw);

bigml = log(mean(exp(shortw-repmat(maxw,M/20,1)),1)) + maxw;
ml = mean(bigml);
mlstd = std(bigml)/sqrt(20);
out = struct('store_w',store_w, 'bigml',bigml);
end
