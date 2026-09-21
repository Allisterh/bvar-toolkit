% bvar.priors.niw - natural-conjugate (normal-inverse-Wishart) prior for a
% VAR(p) with intercept, in four legacy parameterizations.
%
%   [A0,VA0,nu0,S0,U_hat] = bvar.priors.niw(p, kappa, Y0, Y, variant)
%
%   p       : lag length
%   kappa   : shrinkage hyperparameters; length and meaning set by variant
%   Y0, Y   : presample rows and the T x n estimation sample. The univariate
%             AR(4) fits prepend Y0(end-p+1:end,:), whose design matrix is
%             conformable only for p = 4; 'mlvarsv_ncp' prepends the last 4
%             rows whatever p is
%   variant : one of the four below. They are numerically different; never
%             merge them
%     'largebvar_nc'  kappa = [c1 c2], the coefficient and the intercept scale;
%                     VA0 a dense k x 1 vector; nu0 = n+3; S0 = diag(sig2)
%     'mlvarsv_ncp'   as 'largebvar_nc', except for the presample block above
%     'opthyper_ncp'  kappa = [kappa1 ... kappa5]: coefficient scale, lag decay
%                     exponent, intercept scale, then nu0 = kappa(4)+n+1 and
%                     S0 = kappa(5)*diag(sig2). VA0 is a SPARSE k x k diagonal
%                     matrix here and a k x 1 vector in the other variants
%     'kron_script'   as 'largebvar_nc', except S0 = eye(n)
%   A0      : k x n prior mean of the coefficient matrix, zeros, k = 1+n*p
%   VA0     : prior variances of the coefficients; vector or sparse matrix,
%             see variant
%   nu0, S0 : inverse-Wishart degrees of freedom and scale matrix
%   U_hat   : T x n univariate AR(4) residuals, sig2 their mean squares
%
% See:
% Chan, J.C.C. (2020). Large Bayesian Vector Autoregressions. In: P. Fuleky (Eds),
% Macroeconomic Forecasting in the Era of Big Data, 95-125, Springer, Cham
% Chan, J.C.C. (2023). Comparing Stochastic Volatility Specifications for
% Large Bayesian VARs, Journal of Econometrics, 235(2): 1419-1446.
% Chan, J. C. C., L. Jacobi, and D. Zhu (2020). Efficient Selection of
% Hyperparameters in Large Bayesian VARs Using Automatic Differentiation,
% Journal of Forecasting, 39(6): 934-943.
% Chan, J.C.C. (2020). Large Bayesian VARs: A flexible Kronecker error
% covariance structure, Journal of Business and Economic Statistics,
% 38(1), 68-79.

function [A0,VA0,nu0,S0,U_hat] = niw(p,kappa,Y0,Y,variant)
switch variant
    case {'largebvar_nc','mlvarsv_ncp','opthyper_ncp','kron_script'}
    otherwise
        error('bvar:priors:niw:unknownVariant', ...
            'unknown variant ''%s''; use largebvar_nc, mlvarsv_ncp, opthyper_ncp or kron_script', variant);
end
[Tt,n] = size(Y);
k = 1+n*p;
A0 = zeros(k,n);
VA0 = zeros(k,1);
sig2 = zeros(n,1);
U_hat = zeros(Tt,n);
    % construct VA0
if strcmp(variant,'mlvarsv_ncp')
    tmpY = [Y0(end-4+1:end,:); Y];
else
    tmpY = [Y0(end-p+1:end,:); Y];
end
for i=1:n
    Z = [ones(Tt,1) tmpY(4:end-1,i) tmpY(3:end-2,i) tmpY(2:end-3,i)...
        tmpY(1:end-4,i)];
    tmpb = (Z'*Z)\(Z'*tmpY(5:end,i));
    U_hat(:,i) = tmpY(5:end,i)-Z*tmpb;
    sig2(i) = mean(U_hat(:,i).^2);
end
if strcmp(variant,'opthyper_ncp')
    for i=1:k
        l = ceil((i-1)/n);
        idx = mod(i-1,n); % variable index
        if idx==0
            idx = n;
        end
        if i==1 % intercept
            VA0(1) = kappa(3);
        else
            VA0(i) = kappa(1)/(l^kappa(2)*sig2(idx));
        end
    end
    S0 = kappa(5)*diag(sig2); nu0 = kappa(4)+n+1;
    VA0 = sparse(1:k,1:k,VA0);
else
    c1 = kappa(1); c2 = kappa(2);
    for i=1:k
        l = ceil((i-1)/n);
        idx = mod(i-1,n); % variable index
        if idx==0
            idx = n;
        end
        if i==1 % intercept
            VA0(1) = c2;
        else
            VA0(i) = c1/(l^2*sig2(idx));
        end
    end
    nu0 = n+3;
    if strcmp(variant,'kron_script')
        S0 = eye(n);
    else
        S0 = diag(sig2);
    end
end
end
