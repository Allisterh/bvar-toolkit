% bvar.ml.kron_bvar - log marginal likelihood of the standard (homoskedastic,
% natural-conjugate) BVAR of Chan (2020, JBES), in closed form: the identity
% llike + lpri - lpost, with both NIW ordinates exact, evaluated at A = Ahat,
% the posterior mean, and Sig = Shat/(T+nu0), the inverse of the posterior
% mean of inv(Sig). The identity holds at any point, so the choice of point
% affects only rounding. No rng is consumed.
%
%   [ML, out] = bvar.ml.kron_bvar(shortY, X, pri, est)
%
%   pri: A0, VA0 (k x 1), nu0, S0        [replication preset, cited there]
%   est: Ahat, Shat, KA, the posterior mean of A, the inverse-Wishart posterior
%        scale and the posterior precision of each column of A
%   out: llike, and the evaluation point A, Sig
%
% See:
% Chan, J.C.C. (2020). Large Bayesian VARs: A flexible Kronecker error
% covariance structure, Journal of Business and Economic Statistics, 38(1), 68-79.

function [ML, out] = kron_bvar(shortY, X, pri, est)
[T, n] = size(shortY);
k = size(X, 2);
A0 = pri.A0; VA0 = pri.VA0; nu0 = pri.nu0; S0 = pri.S0;
Ahat = est.Ahat; Shat = est.Shat; KA = est.KA;

A = Ahat;
Sig = Shat/(T+nu0);
CSig = chol(Sig,'lower');
tmp = (shortY-X*A)/CSig';
s2 = sum(tmp.^2,2);
llike =  -T*n/2*log(2*pi) - T*sum(log(diag(CSig))) -.5*sum(s2);
ML = llike + bvar.ml.lniwpdf(A,Sig,A0,sparse(1:k,1:k,1./VA0),nu0,S0) ...
    - bvar.ml.lniwpdf(A,Sig,Ahat,KA,nu0+T,Shat);

out = struct('llike', llike, 'A', A, 'Sig', Sig);
end
