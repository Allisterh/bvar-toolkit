% bvar.util.dirirnd - draws from the Dirichlet distribution.
%
%   draws = bvar.util.dirirnd(alp)
%   draws = bvar.util.dirirnd(alp, N)
%
% alp is a COLUMN vector of r positive parameters; a row vector makes gamrnd raise
% 'Size information is inconsistent'. draws is N x r, one draw per row; N defaults
% to 1.
%
% Requires the Statistics and Machine Learning Toolbox (gamrnd).
%
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for
% time-varying parameter VARs with stochastic volatility, Journal of Applied
% Econometrics, 33(4), 509-532.

function draws = dirirnd(alp,N)
if nargin == 1
    N = 1;
end
n = length(alp);
x = gamrnd(repmat(alp',N,1),1,N,n);
draws = x./repmat(sum(x,2),1,n);
end
