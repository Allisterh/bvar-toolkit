% bvar.samplers.eq_var_oi - equation-by-equation Gaussian draw of the VAR
% coefficient matrix A (k x n, intercept first) in the order-invariant VAR-SV
% model
%     y_t = A' x_t + u_t,   B0 u_t = eps_t,   eps_t ~ N(0, diag(exp(h_t))),
% the same conditional as bvar.samplers.eq_svar_oi, computed in O(T k^2 + k^3)
% per equation in place of O(T n k^2 + k^3).
%
%   A = bvar.samplers.eq_var_oi(Y, X, B0, h, A, tmpdV)
%
%   Y     : T x n observations
%   X     : T x k regressors [1, y_{t-1}, ..., y_{t-p}] (bvar.util.build_lags)
%   B0    : n x n INVERSE impact matrix, eps_t = B0 u_t, so the impact matrix
%           itself is B0^{-1} and the reduced-form covariance is
%           B0^{-1} diag(exp(h_t)) B0^{-T}. Pass B0, not B0^{-1}.
%   h     : T x n log-variances of the structural innovations, column j = eps_jt
%   A     : current k x n coefficients (columns updated in place, in order)
%   tmpdV : k*n stacked prior variances, column-major (zero prior mean; to use
%           a prior mean A0, call on Y - X*A0 with A - A0 and add A0 back -
%           an exact reparameterization, tested in tests/unit/test_eq_var_oi.m)
%
% bvar.samplers.eq_svar_oi draws from the same conditional with the same rng
% consumption, randn(k,1) per equation in order, and stays the bitwise anchor
% for replicating Chan, Koop and Yu (2024): it must never be edited for speed.
% Use this function in new code. The "svar" in that name refers to the
% structural parameterization of the error covariance; the coefficients are
% reduced-form in both. Measured ratios, and the shapes they were measured at,
% are tabulated in tests/variant_map.md.
%
% See:
% Chan, J.C.C., Koop, G. and Yu, X. (2024). Large Order-Invariant Bayesian
% VARs with Stochastic Volatility, Journal of Business and Economic
% Statistics, 42(2): 825-837.

function A = eq_var_oi(Y, X, B0, h, A, tmpdV)
[T, n] = size(Y);
k = size(X, 2);
if ~isequal(size(h), [T n])
    error('bvar:samplers:eq_var_oi:badH', ...
        'h must be %d x %d (T x n), not %d x %d', T, n, size(h,1), size(h,2));
end
if ~isequal(size(B0), [n n])
    error('bvar:samplers:eq_var_oi:badB0', ...
        'B0 must be %d x %d, not %d x %d', n, n, size(B0,1), size(B0,2));
end
if numel(tmpdV) ~= k*n
    error('bvar:samplers:eq_var_oi:badV', ...
        'tmpdV must have k*n = %d elements, not %d', k*n, numel(tmpdV));
end
eh_inv = exp(-h);                          % T x n, column j = exp(-h_jt)
for ii = 1:n
    A(:, ii) = 0;
    Etil = (Y - X*A)*B0';                  % row t = (B0 (y_t - A_{-ii}' x_t))'
    b = B0(:, ii);                         % loadings of A(:,ii) in the n structural equations
    w = eh_inv*(b.^2);                     % T x 1 precision weights
    c = (eh_inv.*Etil)*b;                  % T x 1
    iVi = 1./tmpdV((ii-1)*k+1:ii*k);
    Kai = X'*(w.*X);
    Kai(1:k+1:end) = Kai(1:k+1:end) + iVi(:)';
    CKai = chol(Kai, 'lower');
    ai_hat = CKai'\(CKai\(X'*c));
    A(:, ii) = ai_hat + CKai'\randn(k, 1);
end
end
