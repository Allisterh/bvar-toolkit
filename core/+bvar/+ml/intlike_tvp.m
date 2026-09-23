% bvar.ml.intlike_tvp - log integrated likelihood of a TVP-VAR with constant error
% variances, the states theta_t = theta_{t-1} + u_t integrated out analytically.
%
%   llike = bvar.ml.intlike_tvp(Y, Sig, Sigtheta, bigX, theta0)
%
%   Y        : T*n x 1 data, stacked by period
%   Sig      : n x 1 error variances
%   Sigtheta : k x 1 variances of the state innovations
%   bigX     : T*n x T*k regressor matrix of the stacked states
%   theta0   : k x 1 initial state
%
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for
% time-varying parameter VARs with stochastic volatility, Journal of Applied
% Econometrics, 33(4), 509-532.

function llike = intlike_tvp(Y,Sig,Sigtheta,bigX,theta0)
n = size(Sig,1);
Tn = length(Y);
T = Tn/n;
k = size(Sigtheta,1);
Htheta = speye(T*k) - sparse(k+1:T*k,1:(T-1)*k,ones((T-1)*k,1),T*k,T*k);
invSig = sparse(1:T*n,1:T*n,repmat(1./Sig,T,1));
invS = sparse(1:T*k,1:T*k,repmat(1./Sigtheta',1,T),T*k,T*k);
XinvSig = bigX'*invSig;
HinvSH = Htheta'*invS*Htheta;
alptheta = Htheta\[theta0;sparse((T-1)*k,1)];
Ktheta = HinvSH + XinvSig*bigX;
dtheta = XinvSig*Y + HinvSH*alptheta;
CKtheta = chol(Ktheta,'lower');

llike = -T*n/2*log(2*pi) - T/2*sum(log(Sigtheta)) - T/2*sum(log(Sig)) ...
    - sum(log(diag(CKtheta))) - .5*(Y'*invSig*Y ...
    + alptheta'*HinvSH*alptheta - dtheta'*(CKtheta'\(CKtheta\dtheta)));
end
