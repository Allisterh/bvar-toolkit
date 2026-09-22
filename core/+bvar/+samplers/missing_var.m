% bvar.samplers.missing_var - draw the missing values of a VAR(p) in one block, given the
% parameters, subject to linear restrictions M*y = z on the stacked data, hard or soft.
%
%   [ym, ymhat, S] = bvar.samplers.missing_var(Y, A, Sig, h, p)
%   [ym, ymhat, S] = bvar.samplers.missing_var(..., 'M', M, 'z', z, 'V0', 100, 'm0', 0, 'ndraws', 1)
%   [ym, ymhat, S] = bvar.samplers.missing_var(..., 'M', M, 'z', z, 'O', o)
%
%   Y     : T x n data, NaN wherever a value is missing; the first p rows are the
%           initial conditions and may have missing values too
%   A     : k x n VAR coefficients, k = 1 + n*p, intercept first, in the order of
%           bvar.util.build_lags, so that y_t' = x_t'*A + e_t'
%   Sig   : n x n error covariance scale, e_t ~ N(0, exp(h_t)*Sig) for t = p+1, ..., T
%   h     : (T-p) x 1 log-volatilities, or [] for a constant covariance
%   'M', 'z' : q x Tn restriction matrix and q x 1 values on y = vec(Y'), the stacking
%           of bvar.util.select_obs; every row must involve a missing value. Default none
%   'O'   : variances of the errors in soft restrictions z = M*y + e, e ~ N(0, diag(o)),
%           a scalar or q x 1; default [], which makes the restrictions hard, M*y = z
%   'V0', 'm0' : prior variance and mean of the missing values in the first p rows,
%           a scalar or one entry per variable; defaults 100 and 0
%   'ndraws' : number of independent draws, default 1
%   ym    : Nm x ndraws draws of the missing values, in the order of select_obs
%   ymhat : Nm x 1 conditional mean, restrictions included
%   S     : struct with So, Sm and yo from select_obs, so that y = S.So*S.yo + S.Sm*ym
%
% Given the parameters, the missing values are Gaussian with a banded precision matrix K;
% one lower Cholesky factor of K gives the mean, the draw and the update that imposes
% M*y = z exactly. Soft restrictions add Mm'*diag(o)^{-1}*Mm to K, with Mm = M*Sm, and need
% no update. One call consumes Nm*ndraws standard normals. Written for this toolkit.
%
% See:
% Chan, J.C.C., Poon, A. and Zhu, D. (2023). High-Dimensional Conditionally Gaussian
% State Space Models with Missing Data, Journal of Econometrics, 236(1): 105468,
% Sections 2.1-2.2 and Appendix D.
% Rue, H. and Held, L. (2005). Gaussian Markov Random Fields: Theory and Applications,
% Chapman & Hall/CRC, Algorithm 2.6.

function [ym, ymhat, S] = missing_var(Y, A, Sig, h, p, varargin)
M = []; z = []; O = []; V0 = 100; m0 = 0; ndraws = 1;
if mod(numel(varargin), 2) ~= 0
    error('bvar:samplers:missing_var:badOption', 'options must come in name-value pairs');
end
for iv = 1:2:numel(varargin)
    switch lower(char(string(varargin{iv})))
        case 'm',      M = varargin{iv+1};
        case 'z',      z = varargin{iv+1};
        case 'o',      O = varargin{iv+1};
        case 'v0',     V0 = varargin{iv+1};
        case 'm0',     m0 = varargin{iv+1};
        case 'ndraws', ndraws = varargin{iv+1};
        otherwise, error('bvar:samplers:missing_var:badOption', 'unknown option ''%s''', ...
                char(string(varargin{iv})));
    end
end
[T, n] = size(Y);
k = 1 + n*p;
Te = T - p;
if Te < 1 || ~isequal(size(A), [k n]) || ~isequal(size(Sig), [n n])
    error('bvar:samplers:missing_var:badSize', ...
        'Y must have more than p rows, A must be %d x %d and Sig %d x %d', k, n, n, n);
end
if isempty(h), h = zeros(Te, 1); end
if numel(h) ~= Te
    error('bvar:samplers:missing_var:badSize', 'h must have T - p = %d entries', Te);
end
if xor(isempty(M), isempty(z)) || (~isempty(M) && (size(M,2) ~= T*n || size(M,1) ~= numel(z)))
    error('bvar:samplers:missing_var:badRestriction', ...
        'M must be q x %d and z q x 1, given together', T*n);
end
if ~isempty(O) && (isempty(M) || ~(isscalar(O) || numel(O) == numel(z)) ...
        || any(~(O(:) > 0 & isfinite(O(:)))))
    error('bvar:samplers:missing_var:badRestriction', ...
        'O must come with M and z and hold one positive variance, or one per row of M');
end
V0 = V0(:) .* ones(n,1);  m0 = m0(:) .* ones(n,1);

[So, Sm, yo] = bvar.util.select_obs(Y);
S = struct('So', So, 'Sm', Sm, 'yo', yo);
Nm = size(Sm, 2);
if Nm == 0
    ym = zeros(0, ndraws); ymhat = zeros(0, 1);
    return
end

% H*y - 1 kron b0 = e for periods p+1, ..., T, with y_t = b0 + B_1*y_{t-1} + ... + e_t
H = kron(sparse(1:Te, p+1:T, 1, Te, T), speye(n));
for l = 1:p
    Bl = A(1+(l-1)*n+1 : 1+l*n, :)';
    H = H - kron(sparse(1:Te, (p+1:T)-l, 1, Te, T), sparse(Bl));
end
iSig = Sig\eye(n);  iSig = (iSig + iSig')/2;
iOm = kron(sparse(1:Te, 1:Te, exp(-h(:))), sparse(iSig));
Gm = H*Sm;
r = kron(ones(Te,1), A(1,:)') - H*(So*yo);
K = Gm'*iOm*Gm;
c = Gm'*(iOm*r);

% the prior on the missing initial conditions
im = find(isnan(reshape(Y', T*n, 1)));
tm = ceil(im/n);  vm = im - (tm-1)*n;
init = tm <= p;
if any(init)
    K = K + sparse(find(init), find(init), 1./V0(vm(init)), Nm, Nm);
    c(init) = c(init) + m0(vm(init))./V0(vm(init));
end

if ~isempty(M)
    Mm = M*Sm;
    if any(sum(abs(Mm), 2) == 0)
        error('bvar:samplers:missing_var:badRestriction', ...
            'a row of M involves no missing value');
    end
    zm = z(:) - M*(So*yo);
    if ~isempty(O)                                 % soft: z enters as observations
        iO = sparse(1:numel(zm), 1:numel(zm), 1./(O(:).*ones(numel(zm),1)));
        K = K + Mm'*iO*Mm;
        c = c + Mm'*(iO*zm);
    end
end

C = chol(K, 'lower');
ymhat = C'\(C\c);
ym = ymhat + C'\randn(Nm, ndraws);

if ~isempty(M) && isempty(O)                       % hard: the update imposes M*y = z
    U = C'\(C\full(Mm'));
    MU = Mm*U;
    ymhat = ymhat + U*(MU\(zm - Mm*ymhat));
    ym = ym + U*(MU\(zm - Mm*ym));
end
end
