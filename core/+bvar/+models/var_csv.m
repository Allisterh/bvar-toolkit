% bvar.models.var_csv - posterior sampler for a VAR with common stochastic
% volatility, y_t' = x_t'A + u_t' with u_t ~ N(0, exp(h_t)*Sig), h_t a zero-mean
% AR(1) and a natural conjugate prior on (A,Sig). The specification is that of
% Carriero, Clark and Marcellino (2016) as implemented in Chan (2023).
%
%   res = bvar.models.var_csv(Y0, Y, p)
%   res = bvar.models.var_csv(Y0, Y, p, 'nsim', 2000, 'burnin', 500, 'seed', 1)
%
%   Y0 : initial conditions, at least max(p,4) rows, same columns as Y
%   Y  : T x n estimation sample, n >= 2, each series stationary, no NaN or Inf
%   p  : lag length
%
%   'kappa'  : [shrinkage, intercept variance], default [.2^2 100]
%   'nsim'   : draws kept, default 10000;  'burnin' : draws discarded first,
%              default 1000
%   'seed'   : if given, rng(seed,'twister') is set before the first draw
%   'c_reject' : envelope constant of the accept-reject step in bvar.sv.csv_armh,
%              default 3; it changes the cost of the h draw and leaves the
%              target unchanged
%   'draws'  : true also returns res.draws with one row per draw: A (nsim x k*n,
%              each row A(:)'), Sig (nsim x n^2, each row Sig(:)'), h (nsim x T),
%              phi and sigh2; default false
%
%   res.A_mean     : k x n posterior mean of the VAR coefficients, intercept
%                    first, k = 1 + n*p
%   res.Sig_mean   : n x n posterior mean of the scale matrix Sig
%   res.h_mean     : T x 1 posterior mean of the log-volatility
%   res.phi_mean, res.sigh2_mean : posterior means of the AR(1) parameters
%   res.accept_rate : share of sweeps whose Metropolis step accepted, past the
%                    first 20, which take the proposal regardless
%   plus the settings: nsim, burnin, seed, p, c_reject
%
% One factor multiplies the whole covariance matrix, so the order of the columns
% of Y does not affect the volatility model. test_var_csv pins the draws,
% bitwise, to the inline sampler of ex05.
%
% See:
% Carriero, A., Clark, T.E. and Marcellino, M. (2016). Common Drifting Volatility
% in Large Bayesian VARs, Journal of Business and Economic Statistics, 34(3):
% 375-390.
% Chan, J.C.C. (2023). Comparing Stochastic Volatility Specifications for Large
% Bayesian VARs, Journal of Econometrics, 235(2): 1419-1446.

function res = var_csv(Y0, Y, p, varargin)
nsim = 10000; burnin = 1000; seed = []; keep_draws = false;
c_reject = 3; kappa = [.2^2 100];
iv = 1;
while iv <= numel(varargin)
    if iv == numel(varargin)
        error('bvar:models:var_csv:badOption', 'option ''%s'' has no value', char(string(varargin{iv})));
    end
    switch lower(char(string(varargin{iv})))
        case 'kappa',    kappa = varargin{iv+1};
        case 'nsim',     nsim = varargin{iv+1};
        case 'burnin',   burnin = varargin{iv+1};
        case 'seed',     seed = varargin{iv+1};
        case 'c_reject', c_reject = varargin{iv+1};
        case 'draws',    keep_draws = varargin{iv+1};
        otherwise, error('bvar:models:var_csv:badOption', 'unknown option ''%s''', char(string(varargin{iv})));
    end
    iv = iv + 2;
end

[T, n] = size(Y);
if n < 2
    error('bvar:models:var_csv:badData', 'Y must have at least two columns (variables)');
end
if size(Y0, 2) ~= n
    error('bvar:models:var_csv:badData', 'Y0 has %d columns and Y has %d', size(Y0, 2), n);
end
if size(Y0, 1) < max(p, 4)
    error('bvar:models:var_csv:badData', ...
        'Y0 has %d rows; the lags and the AR(4) prior scaling need at least %d', size(Y0, 1), max(p, 4));
end
if ~all(isfinite([Y0(:); Y(:)]))
    error('bvar:models:var_csv:badData', ...
        'Y0 and Y contain NaN or Inf; the sampler has no step for missing values');
end
if ~(isscalar(nsim) && nsim >= 1 && nsim == fix(nsim) && isscalar(burnin) && burnin >= 0 && burnin == fix(burnin))
    error('bvar:models:var_csv:badOption', 'nsim must be a positive integer and burnin a nonnegative integer');
end
if ~isempty(seed), rng(seed, 'twister'); end

k = 1 + n*p;
[~, X] = bvar.util.build_lags([Y0(end-p+1:end,:); Y], p);

Hyper = struct('nuh', 3, 'Sh', .2, 'phi0', .98, 'Vphi', .05^2);
[A0, VA0, nu0, S0] = bvar.priors.niw(p, kappa, Y0, Y, 'mlvarsv_ncp');
iVA0 = sparse(1:k, 1:k, 1./VA0);
VA0iA0 = sparse(1:k, 1:k, VA0)\A0;

phi = Hyper.phi0;
sigh2 = 1/gamrnd(Hyper.nuh, 1/Hyper.Sh);
h = zeros(T,1);

A_sum = zeros(k,n); Sig_sum = zeros(n); h_sum = zeros(T,1);
phi_sum = 0; sigh2_sum = 0;
n_accept = 0; n_mh = 0;
if keep_draws
    D = struct('A', zeros(nsim, k*n), 'Sig', zeros(nsim, n*n), 'h', zeros(nsim, T), ...
        'phi', zeros(nsim,1), 'sigh2', zeros(nsim,1));
end

for isim = 1:nsim + burnin
        % ---- BLOCK 1: (A, Sig) given h, one joint draw ----
    iOh = sparse(1:T, 1:T, exp(-h));
    XiOh = X'*iOh;
    KA = iVA0 + XiOh*X;
    CKA = chol(KA, 'lower');
    Ahat = (CKA')\(CKA\(VA0iA0 + XiOh*Y));
    Shat = S0 + A0'*iVA0*A0 + Y'*iOh*Y - Ahat'*KA*Ahat;
    Shat = (Shat + Shat')/2;                       % symmetrize against rounding
    Sig = iwishrnd(Shat, nu0 + T);
    CSig = chol(Sig, 'lower');
    A = Ahat + (CKA'\randn(k,n))*CSig';
    U = Y - X*A;
    s2 = sum((U/CSig').^2, 2);

        % ---- BLOCK 2: h given the VAR parameters ----
    if isim <= 20
        h = bvar.sv.csv_armh(s2, phi, sigh2, h, n, true, [], 'c_reject', c_reject);
    else
        [h, is_accept] = bvar.sv.csv_armh(s2, phi, sigh2, h, n, false, [], ...
            'c_reject', c_reject);
        n_accept = n_accept + is_accept;
        n_mh = n_mh + 1;
    end

        % ---- BLOCK 3: the AR(1) parameters given h ----
    [phi, sigh2] = bvar.sv.sv0_params(h, phi, Hyper);

    if isim > burnin
        i = isim - burnin;
        A_sum = A_sum + A;
        Sig_sum = Sig_sum + Sig;
        h_sum = h_sum + h;
        phi_sum = phi_sum + phi;
        sigh2_sum = sigh2_sum + sigh2;
        if keep_draws
            D.A(i,:) = A(:)';
            D.Sig(i,:) = Sig(:)';
            D.h(i,:) = h';
            D.phi(i) = phi;
            D.sigh2(i) = sigh2;
        end
    end
end

res.A_mean = A_sum/nsim;
res.Sig_mean = Sig_sum/nsim;
res.h_mean = h_sum/nsim;
res.phi_mean = phi_sum/nsim;
res.sigh2_mean = sigh2_sum/nsim;
res.accept_rate = n_accept/max(n_mh,1);
res.nsim = nsim; res.burnin = burnin; res.seed = seed; res.p = p; res.c_reject = c_reject;
if keep_draws, res.draws = D; end
end
