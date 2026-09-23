% bvar.ml.dirifit - maximum likelihood fit of a Dirichlet distribution to the rows
% of X, by Newton's method started at the column means.
%
%   [alphat, flag] = bvar.ml.dirifit(X)
%
% X is N x k with rows in the simplex; alphat is k x 1. flag is 0 when an iterate
% turns negative, where the fit stops; otherwise 1. The iteration stops once any
% element of the Newton step is at most 1e-4 in absolute value. This is the
% published stopping rule, kept deliberately: the importance densities of the RS
% marginal likelihoods depend on it.
%
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for
% time-varying parameter VARs with stochastic volatility, Journal of Applied
% Econometrics, 33(4), 509-532.

function [alphat,flag] = dirifit(X)
[N,~] = size(X);
alphat = mean(X)';
logX = log(X);
err = 1;
flag = 1;
while abs(err) > 1e-4 % stopping criteria
    if min(alphat) < 0
        flag = 0;
        break
    else
        tmpS = repmat(psi(sum(alphat)) - psi(alphat)',N,1) + logX;
        S = sum(tmpS)';
        H = N*psi(1,sum(alphat)) - N*diag(psi(1,alphat));
        err = - H\S;
        alphat = alphat + err ;
    end
end
end
