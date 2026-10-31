% bvar.structural.qr_sign - QR decomposition with the diagonal of R normalized
% to be positive, which makes the decomposition unique and gives the orthogonal
% factor Q used to draw rotations in sign-restricted SVARs.
%
%   [Q,R] = bvar.structural.qr_sign(A)
%
% For A = randn(n,n), Q is drawn from the uniform (Haar) measure on the
% orthogonal group. Callers use it to rotate the Cholesky factor of the
% reduced-form covariance.
%
% See:
% Chan, J.C.C. (2022). Asymmetric Conjugate Priors for Large Bayesian VARs,
% Quantitative Economics, 13(3): 1145-1169.

function [Q,R] = qr_sign(A)
m = size(A,1);
[Q,R] = qr(A);
Q = Q*sparse(1:m,1:m,sign(diag(R)));
R = sparse(1:m,1:m,sign(diag(R)))*R;
end
