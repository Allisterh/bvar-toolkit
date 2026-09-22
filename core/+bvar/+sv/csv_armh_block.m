% bvar.sv.csv_armh_block - one sweep of accept-reject Metropolis-Hastings updates of the
% common stochastic volatility path h, one block of consecutive periods at a time.
%
%   [h, n_accept, n_block] = bvar.sv.csv_armh_block(s2, rho, sigh2, h, n)
%   [h, n_accept, n_block] = bvar.sv.csv_armh_block(..., 'block', 60, 'c_reject', 3)
%
%   s2          : T x 1, e_t'*Sig^{-1}*e_t at each t
%   rho, sigh2  : AR(1) coefficient and innovation variance of the zero-mean h, whose
%                 first value has the stationary distribution
%   h, n        : current log-volatility path (T x 1) and number of series
%   'block'     : block length, default 60; unless 'block' >= T, the first block has a
%                 random length between 1 and 'block', drawn with randi
%   'c_reject'  : c in the envelope target <= c * proposal, default 3
%   'ForcedAccept' : take every proposal regardless of the MH ratio, default false
%   'MaxIterMode', 'MaxIterAR' : caps on the two loops, defaults 500 and 1000
%   n_accept, n_block : blocks whose proposal was accepted, and blocks in the sweep
%
% Each block is drawn from its conditional given the rest of the path, with the proposal
% of bvar.sv.csv_armh, a Gaussian at the mode of that conditional. With 'block' >= T the
% path is one block, and the update differs from bvar.sv.csv_armh only in evaluating the
% proposal precision at the mode. Written for this toolkit.
%
% See:
% Chan, J.C.C. (2017). The Stochastic Volatility in Mean Model with Time-Varying
% Parameters: An Application to Inflation Modeling, Journal of Business and Economic
% Statistics, 35(1): 17-28.
% Chan, J.C.C. (2020). Large Bayesian VARs: A Flexible Kronecker Error Covariance
% Structure, Journal of Business and Economic Statistics, 38(1): 68-79.

function [h, n_accept, n_block] = csv_armh_block(s2, rho, sigh2, h, n, varargin)
L = 60; c_reject = 3; forced = false; maxit_mode = 500; maxit_ar = 1000;
tol_mode = 1e-3;
if mod(numel(varargin), 2) ~= 0
    error('bvar:sv:csv_armh_block:badOption', 'options must come in name-value pairs');
end
for iv = 1:2:numel(varargin)
    switch lower(char(string(varargin{iv})))
        case 'block',        L = varargin{iv+1};
        case 'c_reject',     c_reject = varargin{iv+1};
        case 'forcedaccept', forced = varargin{iv+1};
        case 'maxitermode',  maxit_mode = varargin{iv+1};
        case 'maxiterar',    maxit_ar = varargin{iv+1};
        otherwise, error('bvar:sv:csv_armh_block:badOption', 'unknown option ''%s''', ...
                char(string(varargin{iv})));
    end
end
if ~(isscalar(L) && L >= 1 && L == fix(L))
    error('bvar:sv:csv_armh_block:badOption', 'block must be a positive integer');
end
if ~(isscalar(c_reject) && isreal(c_reject) && isfinite(c_reject) && c_reject > 0)
    error('bvar:sv:csv_armh_block:badOption', 'c_reject must be a finite positive scalar');
end

s2 = s2(:);  h = h(:);
T = numel(s2);
Hrho = speye(T) - rho*sparse(2:T, 1:T-1, 1, T, T);
Q = Hrho'*sparse(1:T, 1:T, [(1-rho^2)/sigh2; ones(T-1,1)/sigh2])*Hrho;

if L >= T
    edges = [0 T];
else
    first = randi(L);
    edges = unique([0, first:L:T, T]);
end
n_block = numel(edges) - 1;
n_accept = 0;
for ib = 1:n_block
    B = (edges(ib)+1 : edges(ib+1))';
    QBB = Q(B,B);
    b = Q(B,:)*h - QBB*h(B);                     % the terms linking the block to the rest
    sB = s2(B);
    logpi = @(x) -.5*x'*QBB*x - x'*b - n/2*sum(x) - .5*exp(-x)'*sB;

    x = h(B);  err = Inf;  it = 0;
    while ~(err <= tol_mode)
        it = it + 1;
        if it > maxit_mode
            error('bvar:sv:csv_armh_block:modeNotConverged', ...
                'mode search did not converge in %d iterations (last max|dh| = %g)', maxit_mode, err);
        end
        G = .5*sB.*exp(-x);
        K = QBB + sparse(1:numel(B), 1:numel(B), G);
        xnew = K\(G.*x - n/2 + G - b);
        err = max(abs(xnew - x));
        x = xnew;
    end
    G = .5*sB.*exp(-x);
    K = QBB + sparse(1:numel(B), 1:numel(B), G);
    CK = chol(K, 'lower');
    logc = logpi(x) + log(c_reject);

    % accept-reject step
    it = 0;
    while true
        it = it + 1;
        if it > maxit_ar
            error('bvar:sv:csv_armh_block:arNotAccepted', ...
                'no accept-reject proposal accepted in %d draws at c_reject = %g', maxit_ar, c_reject);
        end
        xc = x + CK'\randn(numel(B), 1);
        alpARc = logpi(xc) + .5*(xc-x)'*K*(xc-x) - logc;
        if alpARc > log(rand), break, end
    end
    % Metropolis-Hastings step
    alpAR = logpi(h(B)) + .5*(h(B)-x)'*K*(h(B)-x) - logc;
    if alpAR < 0
        alpMH = 0;
    elseif alpARc < 0
        alpMH = -alpAR;
    else
        alpMH = alpARc - alpAR;
    end
    if alpMH > log(rand) || forced
        h(B) = xc;
        n_accept = n_accept + 1;
    end
end
end
