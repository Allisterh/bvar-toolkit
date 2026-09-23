% bvar.ml.ldiripdf - log density of the Dirichlet distribution at each row of y.
%
%   lden = bvar.ml.ldiripdf(y, alpha)
%
% y is N x k, each row a point in the simplex; alpha has k elements, as a row or
% a column; lden is N x 1.
%
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for
% time-varying parameter VARs with stochastic volatility, Journal of Applied
% Econometrics, 33(4), 509-532.

function lden = ldiripdf(y, alpha)
[~,k] = size(y);
if ~(k == length(alpha))
    error('dimensions do not match ');
end
if size(alpha, 1) < size(alpha, 2)
    alpha = alpha';
end

const = gammaln(sum(alpha)) - sum(gammaln(alpha));
lden = const + log(y)*(alpha - 1);
end
