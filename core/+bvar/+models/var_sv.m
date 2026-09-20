% bvar.models.var_sv - posterior sampler for a VAR with stochastic volatility,
% under the Cholesky or the order-invariant specification of the error
% covariance matrix, with the prior of Chan, Koop and Yu (2024).
%
%   res = bvar.models.var_sv(Y0, Y, p)
%   res = bvar.models.var_sv(Y0, Y, p, 'model', 'CS', 'nsim', 1000, 'burnin', 200, 'seed', 1)
%
%   Y0 : initial conditions, at least max(p,4) rows, same columns as Y
%   Y  : T x n estimation sample, n >= 2, each series stationary, no NaN or Inf
%   p  : lag length
%
%   'model'  : 'OI' (default), Sigma_t^{-1} = B0' D_t^{-1} B0 with B0 unrestricted
%              and zero-mean log-volatilities; or 'CS', the same with B0 unit
%              lower triangular and log-volatility means estimated
%   'nsim'   : posterior draws kept, default 30000
%   'burnin' : draws discarded first, default 5000
%   'seed'   : if given, rng(seed,'twister') is set before the first draw;
%              otherwise the current stream is used
%   'draws'  : true also returns res.draws, the parameter draws, one row per
%              draw: kappa (nsim x 2), A (nsim x k*n, each row A(:)'), impact
%              (nsim x n^2), phi and sig2 (nsim x n), h_T (nsim x n, the
%              log-volatilities at the end of the sample, which a forecast
%              continues from) and, under 'CS', mu (nsim x n); default false.
%              The rest of the log-volatility paths are not kept.
%   'phi_proposal' : under 'OI', the candidate of the Metropolis-Hastings step for
%              the log-volatility persistence phi, passed to bvar.sv.sv0_params:
%              'truncated' (default) or 'untruncated', the step of the Chan, Koop
%              and Yu (2024) package
%
%   res.Sig_mean     : T x n x n posterior mean of Sigma_t
%   res.A_mean       : k x n posterior mean of the VAR coefficients, intercept
%                      first, k = 1 + n*p
%   res.h_mean       : T x n posterior mean of the log-volatilities
%   res.impact_mean  : n x n posterior mean of B0 ('OI') or of the unit lower
%                      triangular matrix ('CS')
%   res.kappa_mean   : posterior means of the own- and cross-lag shrinkage
%   plus the settings: model, nsim, burnin, seed, p
%
% The prior constants and the chain initialization are those of the Chan, Koop
% and Yu (2024) package (replications/chan_koop_yu2024_jbes_oisv/preset.m).
% test_var_sv checks the constants against that file and pins the draws, bitwise,
% to the inline sampler ex06 used previously, whose phi step is the package's
% (under 'OI', 'phi_proposal' set to 'untruncated').
%
% See:
% Cogley, T. and Sargent, T.J. (2005). Drifts and Volatilities: Monetary
% Policies and Outcomes in the Post WWII US, Review of Economic Dynamics,
% 8(2): 262-302.
% Chan, J.C.C., Koop, G. and Yu, X. (2024). Large Order-Invariant Bayesian
% VARs with Stochastic Volatility, Journal of Business and Economic
% Statistics, 42(2): 825-837.

function res = var_sv(Y0, Y, p, varargin)
model = 'OI'; nsim = 30000; burnin = 5000; seed = []; keep_draws = false;
phi_proposal = 'truncated';
iv = 1;
while iv <= numel(varargin)
    if iv == numel(varargin)
        error('bvar:models:var_sv:badOption', 'option ''%s'' has no value', char(string(varargin{iv})));
    end
    switch lower(char(string(varargin{iv})))
        case 'model',  model  = upper(char(string(varargin{iv+1})));
        case 'nsim',   nsim   = varargin{iv+1};
        case 'burnin', burnin = varargin{iv+1};
        case 'seed',   seed   = varargin{iv+1};
        case 'draws',  keep_draws = varargin{iv+1};
        case 'phi_proposal', phi_proposal = lower(char(string(varargin{iv+1})));
        otherwise, error('bvar:models:var_sv:badOption', 'unknown option ''%s''', char(string(varargin{iv})));
    end
    iv = iv + 2;
end

[T, n] = size(Y);
if ~any(strcmp(model, {'OI', 'CS'}))
    error('bvar:models:var_sv:badModel', 'model must be ''OI'' or ''CS''; got ''%s''', model);
end
if n < 2
    error('bvar:models:var_sv:badData', 'Y must have at least two columns (variables)');
end
if size(Y0, 2) ~= n
    error('bvar:models:var_sv:badData', 'Y0 has %d columns and Y has %d', size(Y0, 2), n);
end
if size(Y0, 1) < max(p, 4)
    error('bvar:models:var_sv:badData', ...
        'Y0 has %d rows; the lags and the AR(4) prior scaling need at least %d', size(Y0, 1), max(p, 4));
end
if ~all(isfinite([Y0(:); Y(:)]))
    error('bvar:models:var_sv:badData', ...
        'Y0 and Y contain NaN or Inf; the sampler has no step for missing values');
end
if ~(isscalar(nsim) && nsim >= 1 && nsim == fix(nsim) && isscalar(burnin) && burnin >= 0 && burnin == fix(burnin))
    error('bvar:models:var_sv:badOption', 'nsim must be a positive integer and burnin a nonnegative integer');
end
if ~(isscalar(keep_draws) && (islogical(keep_draws) || isnumeric(keep_draws)))
    error('bvar:models:var_sv:badOption', 'draws must be true or false');
end
if ~any(strcmp(phi_proposal, {'truncated', 'untruncated'}))
    error('bvar:models:var_sv:badOption', ...
        'phi_proposal must be ''truncated'' or ''untruncated''; got ''%s''', phi_proposal);
end
if ~isempty(seed), rng(seed, 'twister'); end

is_oi = strcmp(model, 'OI');
k = 1 + n*p;
[~, X] = bvar.util.build_lags([Y0(end-p+1:end,:); Y], p);

    % constants of the package's preset.m
ls_ridge = .01;                              % chain-init least squares
sv_offset = .0001;                           % ystar = log(E.^2 + sv_offset)
phi_init_bnd = .99;                          % chain-init bound on phi
if is_oi, kappa = [.1 .1 NaN 100]; else, kappa = [.1 .1 1 100]; end

    % Minnesota second moments, built from AR(4) residual variances
sig2_ar = bvar.priors.resid_var_ar4(Y0, Y);
[C, idx_kappa1, idx_kappa2] = bvar.priors.minnesota_C(n, p, sig2_ar);

Hyper.nuh = 3*ones(n,1);
Hyper.Sh = .05*(Hyper.nuh - 1);
Hyper.phi0 = .95*ones(n,1);
Hyper.Vphi = .05^2*ones(n,1);
Hyper.mu0 = zeros(n,1);
Hyper.Vmu = 100*ones(n,1);
Hyper.B0 = eye(n);
Hyper.VB0 = ones(n);
Hyper.beta0 = zeros(n^2*p + n, 1);
Hyper.Valp = ones(n*(n-1)/2, 1);

    % chain init
A = (X'*X + ls_ridge*speye(k))\(X'*Y);
U = Y - X*A;
Sig_hat = U'*U/T;
h = repmat(log(diag(Sig_hat))', T, 1);
sig2 = 1./gamrnd(Hyper.nuh, 1./Hyper.Sh);
phi = min(Hyper.phi0 + sqrt(Hyper.Vphi).*randn(n,1), phi_init_bnd);
z_psi1 = 1./gamrnd(.5, 1, n*p, 1);
z_psi2 = 1./gamrnd(.5, 1, (n-1)*n*p, 1);
z_kappa = 1./gamrnd(.5, 1, 2, 1);
Psi = ones(k*n, 1);
Psi(idx_kappa1) = 1./gamrnd(.5, z_psi1);
Psi(idx_kappa2) = 1./gamrnd(.5, z_psi2);
if is_oi
    B0 = diag(1./sqrt(diag(Sig_hat)));       % full matrix, updated row by row
else
    A_id = nonzeros(tril(reshape(1:n^2, n, n), -1)');
    Atri = eye(n); B = A'; XB = X*B';        % unit lower triangular impact matrix
    mu = zeros(n,1);
    for ii = 1:n, mu(ii) = mean(log(U(:,ii).^2)); end
end

Sig_sum = zeros(T, n, n);
A_sum = zeros(k, n);
h_sum = zeros(T, n);
imp_sum = zeros(n, n);
store_kappa = zeros(nsim, 2);
if keep_draws
    D.kappa = [];                            % store_kappa, filled in at the end
    D.A = zeros(nsim, k*n);
    D.impact = zeros(nsim, n^2);
    D.phi = zeros(nsim, n);
    D.sig2 = zeros(nsim, n);
    D.h_T = zeros(nsim, n);
    if ~is_oi, D.mu = zeros(nsim, n); end
end
for isim = 1:nsim + burnin
        % Vbeta scales with sig2, which holds the log-volatility state variances
        % at this point, as in the published samplers
    [~, tmpdV] = bvar.priors.vtheta(idx_kappa1, idx_kappa2, kappa, C.*Psi, sig2);

    if is_oi
        B0 = bvar.structural.b0_row_sampler(Y - X*A, h, B0, Hyper.B0, Hyper.VB0);
        A = bvar.samplers.eq_var_oi(Y, X, B0, h, A, tmpdV);
        theta = A(:);
        E = (Y - X*A)*B0';                   % structural innovations
    else
        [B, XB] = bvar.samplers.eq_tri_cs(Y, X, XB, B, Atri, h, tmpdV, Hyper.beta0);
        theta = reshape(B', n^2*p + n, 1);
        E = Y - XB;
        Atri(A_id) = bvar.samplers.alp_tri_cs(E, h, Hyper.Valp);
        E = E*sparse(Atri');                 % structural innovations
    end

    for ii = 1:n
        ystar = log(E(:,ii).^2 + sv_offset);
        if is_oi
            h(:,ii) = bvar.sv.ksc_ar1_mean(ystar, h(:,ii), 0, phi(ii), sig2(ii));
        else
            h(:,ii) = bvar.sv.ksc_ar1_mean(ystar, h(:,ii), mu(ii), phi(ii), sig2(ii));
        end
    end
    if is_oi
        [phi, sig2] = bvar.sv.sv0_params(h, phi, Hyper, [], 'proposal', phi_proposal);
    else
        [mu, phi, sig2] = bvar.sv.sv_params(h, mu, phi, Hyper);
    end

    [psi1, psi2, z_psi1, z_psi2, kappa, z_kappa] = bvar.samplers.horseshoe_kappa_psi( ...
        theta, idx_kappa1, idx_kappa2, C, kappa, z_psi1, z_psi2, z_kappa);
    Psi(idx_kappa1) = psi1;
    Psi(idx_kappa2) = psi2;

    if isim > burnin
        store_kappa(isim - burnin, :) = kappa(1:2);
        h_sum = h_sum + h;
        if is_oi
            Sig_sum = Sig_sum + bvar.structural.construct_Sigt(h, B0);
            A_sum = A_sum + A;
            imp_sum = imp_sum + B0;
        else
            Sig_sum = Sig_sum + bvar.structural.construct_Sigt(h, Atri);
            A_sum = A_sum + B';
            imp_sum = imp_sum + Atri;
        end
        if keep_draws
            r = isim - burnin;
            if is_oi
                D.A(r,:) = A(:)';
                D.impact(r,:) = B0(:)';
            else
                D.A(r,:) = reshape(B', 1, []);
                D.impact(r,:) = Atri(:)';
                D.mu(r,:) = mu';
            end
            D.phi(r,:) = phi';
            D.sig2(r,:) = sig2';
            D.h_T(r,:) = h(end,:);
        end
    end
end

res.Sig_mean = Sig_sum/nsim;
res.A_mean = A_sum/nsim;
res.h_mean = h_sum/nsim;
res.impact_mean = imp_sum/nsim;
res.kappa_mean = mean(store_kappa)';
res.model = model; res.nsim = nsim; res.burnin = burnin; res.seed = seed; res.p = p;
if keep_draws
    D.kappa = store_kappa;
    res.draws = D;
end
end
