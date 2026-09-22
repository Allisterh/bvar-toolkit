% bvar.samplers.var_coef_joint - joint draw of all VAR coefficients under an independent
% normal prior with mean zero, given the error covariance and a volatility weight for
% each period.
%
%   A = bvar.samplers.var_coef_joint(Y, X, Sig, w, iVb)
%   [A, Ahat] = bvar.samplers.var_coef_joint(Y, X, Sig, w, iVb)
%
%   Y, X : T x n left-hand side and T x k regressors, as bvar.util.build_lags returns them
%   Sig  : n x n error covariance scale, e_t ~ N(0, Sig/w_t)
%   w    : T x 1 weights, exp(-h_t) under common stochastic volatility
%   iVb  : kn x kn prior precision of vec(A), sparse or full
%   A    : k x n draw, y_t' = x_t'*A + e_t'
%   Ahat : k x n posterior mean, computed only when asked for
%
% The posterior precision of vec(A) is iVb + kron(Sig^{-1}, X'*W*X), W = diag(w); one lower
% Cholesky factor gives the draw. One call consumes kn standard normals. Written for this
% toolkit.

function [A, Ahat] = var_coef_joint(Y, X, Sig, w, iVb)
[k, n] = deal(size(X,2), size(Y,2));
XW = X.*w(:);
iSig = Sig\eye(n);  iSig = (iSig + iSig')/2;
Kb = iVb + kron(sparse(iSig), sparse(XW'*X));
Cb = chol(Kb, 'lower');
cb = reshape(XW'*Y*iSig, k*n, 1);
A = reshape(Cb'\(Cb\cb + randn(k*n,1)), k, n);
if nargout > 1
    Ahat = reshape(Cb'\(Cb\cb), k, n);
end
end
