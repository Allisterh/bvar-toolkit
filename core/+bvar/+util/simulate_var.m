% bvar.util.simulate_var - simulates T observations of a VAR with Gaussian errors, starting from
% its stationary distribution or from given initial values. Written for this toolkit.
%
%   [Y, Y0] = bvar.util.simulate_var(A, Sig, T)
%   [Y, Y0] = bvar.util.simulate_var(A, Sig, T, Y0)
%
%   A   : k x n coefficients, intercepts first, then the lag-1 block, ..., the lag-p block, with
%         k = 1 + n*p (the layout of bvar.util.build_lags)
%   Sig : n x n error covariance matrix, positive definite
%   T   : number of observations to simulate
%   Y0  : p x n initial values y_{1-p}, ..., y_0, oldest first. Given as an input, the path starts
%         from them; otherwise they are drawn from the stationary distribution, which needs a
%         stable VAR, and [Y0; Y] is then a sample of the stationary process
%   Y   : T x n simulated observations y_1, ..., y_T
%
% The draws come from randn: n*p for the initial values when they are drawn, then n per period.
%
% See:
% Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and Applications. Chapman &
% Hall/CRC.

function [Y, Y0] = simulate_var(A, Sig, T, Y0)
[k, n] = size(A);
p = (k - 1)/n;
if p < 1 || p ~= round(p)
    error('simulate_var:badA', 'A must have 1 + n*p rows for its n = %d columns', n);
end
if nargin < 4
    F = [A(2:end, :)'; eye(n*(p-1)), zeros(n*(p-1), n)];   % the companion matrix
    if max(abs(eig(F))) >= 1
        error('simulate_var:unstable', ['the VAR is not stable, so it has no stationary ' ...
            'distribution; pass initial values Y0']);
    end
    c = [A(1, :)'; zeros(n*(p-1), 1)];
    mu = (eye(n*p) - F)\c;                                  % mean of (y_t', ..., y_{t-p+1}')'
    V = blkdiag(Sig, zeros(n*(p-1)));                       % V = sum_j F^j Q F^j', by doubling
    Fj = F;
    for it = 1:200
        Vnew = V + Fj*V*Fj';
        Fj = Fj*Fj;
        done = max(abs(Vnew(:) - V(:))) <= 1e-15*max(abs(Vnew(:)));
        V = Vnew;
        if done, break, end
    end
    s = mu + chol((V + V')/2, 'lower')*randn(n*p, 1);       % (y_0', y_{-1}', ..., y_{1-p}')'
    Y0 = flipud(reshape(s, n, p)');
elseif ~isequal(size(Y0), [p, n])
    error('simulate_var:badY0', 'Y0 must be p x n = %d x %d', p, n);
end
C = chol(Sig, 'lower');
Yfull = [Y0; zeros(T, n)];
for t = p+1:p+T
    x = [1, reshape(Yfull(t-1:-1:t-p, :)', 1, [])];       % (1, y_{t-1}', ..., y_{t-p}')
    Yfull(t, :) = x*A + (C*randn(n, 1))';
end
Y = Yfull(p+1:end, :);
end
