% bvar.ml.intlike_rsvar - log likelihood of a regime-switching VAR with Markov
% regimes, the regimes integrated out by the Hamilton filter.
%
%   llike = bvar.ml.intlike_rsvar(shortY, bigX, theta, Sig, P)
%   llike = bvar.ml.intlike_rsvar(shortY, bigX, theta, Sig, P, p1)
%
%   shortY : T x n data
%   bigX   : T*n x k regressor matrix, bigX*theta(:,j) the stacked conditional
%            mean in regime j
%   theta  : k x r coefficients, one column per regime
%   Sig    : n x r error variances, one column per regime
%   P      : r x r transition matrix, P(i,j) = p(s_t = j | s_{t-1} = i)
%   p1     : p(s_1 = j), the same for every j; default 1/r. The published routine
%            uses 1/3 whatever r is, which shifts the log likelihood by log(r/3);
%            p1 = 1/3 reproduces it.
%
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for
% time-varying parameter VARs with stochastic volatility, Journal of Applied
% Econometrics, 33(4), 509-532.

function llike = intlike_rsvar(shortY,bigX,theta,Sig,P,p1)
r = size(theta,2);
[T,n] = size(shortY);
if nargin < 6
    p1 = 1/r;
end
llike = 0;
like = zeros(T,r);
tmpP1 = zeros(T,r);                   % p(s_t|Y_t,\theta,P)
tmpP2 = zeros(T,r); tmpP2(1,:) = p1;  % p(s_t|Y_{t-1},\theta,P)

    % compute filtering probabilities
for i=1:r
    mu_i = bigX*theta(:,i);
    Sig_i = Sig(:,i);
    like(:,i) = mvnpdf(shortY,reshape(mu_i,n,T)',Sig_i');
end
for t=1:T
    if t>1
        tmpP2(t,:) = tmpP1(t-1,:)*P;
    end
    tmp  = tmpP2(t,:) .* like(t,:) ;
    tmpP1(t,:) = tmp;
    tmpP1(t,:) = tmpP1(t,:)/sum(tmpP1(t,:));
    llike = llike + log(sum(tmp));
end
end
