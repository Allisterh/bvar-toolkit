% bvar.models.mfvar_csv - posterior sampler for a VAR with common stochastic volatility and
% missing data, such as a mixed-frequency VAR, with the independent priors of Chan, Poon
% and Zhu (2023).
%
%   res = bvar.models.mfvar_csv(Y, p)
%   res = bvar.models.mfvar_csv(Y, p, 'M', M, 'z', z, 'sig2', s2, 'nsim', 10000, 'seed', 1)
%
%   Y  : T x n data, NaN wherever a value is missing; the first p rows are the initial
%        conditions. A mixed-frequency VAR has the quarterly series missing in every month
%   p  : lag length
%
%   'M', 'z'  : linear restrictions M*y = z on y = vec(Y'), such as the quarterly
%               aggregation of bvar.util.mm_constraint; default none
%   'sig2'    : n-vector of the variances s_i^2 that scale the Minnesota prior; a NaN
%               entry is replaced by the residual variance of an AR(4) fitted to the
%               observed values of that series, which a series missing in every period
%               does not have, so its entry must be given
%   'kappa'   : [kappa_1 kappa_2], own-lag and other-lag shrinkage, default [.04 .01]
%   'V0', 'm0': prior variance and mean of the missing initial conditions, defaults 100, 0
%   'nsim'    : draws kept, default 10000;  'burnin' : draws discarded first, default 1000
%   'seed'    : if given, rng(seed,'twister') is set before the first draw
%   'hblock'  : block length of the h update, default 36
%   'c_reject': envelope constant of the h update, default 3
%   'quantiles' : probabilities of the reported quantiles, default [.05 .16 .5 .84 .95]
%   'draws'   : true also returns res.draws: ym (nsim x Nm, single), A (nsim x k*n),
%               Sig (nsim x n^2), h (nsim x T-p), phi and sigh2; default false
%
%   res.Y_mean    : T x n data with each missing value replaced by its posterior mean
%   res.Y_q       : T x n x numel(quantiles) quantiles, equal to the data where observed
%   res.A_mean    : k x n posterior mean of the VAR coefficients, intercept first, k = 1+n*p
%   res.Sig_mean, res.h_mean (T-p x 1), res.phi_mean, res.sigh2_mean : posterior means
%   res.accept_rate : share of the blocks of h whose Metropolis step accepted, past the
%               first 20 sweeps, which take the proposal regardless
%   res.max_resid : largest |M*y - z| over the kept draws
%   res.sig2  : the Minnesota scales used;  res.prior.V : k x n prior variances of A;
%               plus the settings and the rest of res.prior
%
% The model is y_t = b_0 + B_1 y_{t-1} + ... + B_p y_{t-p} + e_t with
% e_t ~ N(0, exp(h_t)*Sig) and h_t a zero-mean AR(1). Priors: the coefficients independent
% normal with mean zero and variances kappa_1/l^2 on own lags, kappa_2*s_i^2/(l^2*s_j^2)
% on lag l of variable j in equation i, and 100*s_i^2 on the intercepts;
% Sig ~ IW(n+3, I_n); sigh2 ~ IG(10, .004); phi ~ N(.98, .05^2) on |phi| < 1. Chan, Poon
% and Zhu (2023) give phi this truncated normal prior without its mean and variance; the
% values here are those of bvar.models.var_csv. Each sweep draws the missing values
% (bvar.samplers.missing_var), the coefficients jointly (bvar.samplers.var_coef_joint),
% Sig, h (bvar.sv.csv_armh_block) and (phi, sigh2) (bvar.sv.sv0_params, truncated
% candidate). The draws of h, here in blocks, and of (phi, sigh2) follow the algorithm of
% Chan (2020). Written for this toolkit.
%
% See:
% Chan, J.C.C., Poon, A. and Zhu, D. (2023). High-Dimensional Conditionally Gaussian
% State Space Models with Missing Data, Journal of Econometrics, 236(1): 105468,
% Section 4.1.
% Carriero, A., Clark, T.E. and Marcellino, M. (2016). Common Drifting Volatility in
% Large Bayesian VARs, Journal of Business and Economic Statistics, 34(3): 375-390.
% Chan, J.C.C. (2020). Large Bayesian VARs: A Flexible Kronecker Error Covariance
% Structure, Journal of Business and Economic Statistics, 38(1): 68-79.

function res = mfvar_csv(Y, p, varargin)
M = []; z = []; sig2 = []; kappa = [.04 .01]; V0 = 100; m0 = 0;
nsim = 10000; burnin = 1000; seed = []; c_reject = 3; hblock = 36; keep_draws = false;
probs = [.05 .16 .5 .84 .95];
iv = 1;
while iv <= numel(varargin)
    if iv == numel(varargin)
        error('bvar:models:mfvar_csv:badOption', 'option ''%s'' has no value', char(string(varargin{iv})));
    end
    switch lower(char(string(varargin{iv})))
        case 'm',         M = varargin{iv+1};
        case 'z',         z = varargin{iv+1};
        case 'sig2',      sig2 = varargin{iv+1};
        case 'kappa',     kappa = varargin{iv+1};
        case 'v0',        V0 = varargin{iv+1};
        case 'm0',        m0 = varargin{iv+1};
        case 'nsim',      nsim = varargin{iv+1};
        case 'burnin',    burnin = varargin{iv+1};
        case 'seed',      seed = varargin{iv+1};
        case 'c_reject',  c_reject = varargin{iv+1};
        case 'hblock',    hblock = varargin{iv+1};
        case 'quantiles', probs = varargin{iv+1};
        case 'draws',     keep_draws = varargin{iv+1};
        otherwise, error('bvar:models:mfvar_csv:badOption', 'unknown option ''%s''', char(string(varargin{iv})));
    end
    iv = iv + 2;
end

[T, n] = size(Y);
if n < 2
    error('bvar:models:mfvar_csv:badData', 'Y must have at least two columns (variables)');
end
if T <= p + 4
    error('bvar:models:mfvar_csv:badData', 'Y has %d rows; it needs more than p + 4', T);
end
if any(isinf(Y(:)))
    error('bvar:models:mfvar_csv:badData', 'Y contains Inf; mark missing values with NaN');
end
if ~(isscalar(nsim) && nsim >= 1 && nsim == fix(nsim) && isscalar(burnin) && burnin >= 0 && burnin == fix(burnin))
    error('bvar:models:mfvar_csv:badOption', 'nsim must be a positive integer and burnin a nonnegative integer');
end
if isempty(sig2), sig2 = nan(n,1); end
sig2 = sig2(:);
if numel(sig2) ~= n
    error('bvar:models:mfvar_csv:badOption', 'sig2 must have one entry per variable');
end
for i = find(isnan(sig2))'
    sig2(i) = ar4_resid_var(Y(:,i));
end
if any(isnan(sig2))
    error('bvar:models:mfvar_csv:badOption', ...
        'sig2 must be given for series %s, which have too few observed values for an AR(4)', ...
        mat2str(find(isnan(sig2))'));
end
if ~isempty(seed), rng(seed, 'twister'); end

% priors
k = 1 + n*p;
V = zeros(k, n);                                  % prior variances, one column per equation
for i = 1:n
    V(1,i) = 100*sig2(i);
    for l = 1:p
        for j = 1:n
            if i == j
                V(1+(l-1)*n+j, i) = kappa(1)/l^2;
            else
                V(1+(l-1)*n+j, i) = kappa(2)*sig2(i)/(l^2*sig2(j));
            end
        end
    end
end
iVb = sparse(1:k*n, 1:k*n, 1./V(:));
nu0 = n + 3;  S0 = eye(n);
Hyper = struct('nuh', 10, 'Sh', .004, 'phi0', .98, 'Vphi', .05^2);
Te = T - p;

% starting values: coefficients at the observed means, the missing values at their
% conditional mean under those coefficients
mu = mean(Y, 1, 'omitnan');  mu(isnan(mu)) = 0;
A = [mu; zeros(n*p, n)];
Sig = diag(sig2);
h = zeros(Te, 1);  phi = Hyper.phi0;  sigh2 = Hyper.Sh/(Hyper.nuh - 1);
[~, ymhat, Sel] = bvar.samplers.missing_var(Y, A, Sig, h, p, 'M', M, 'z', z, 'V0', V0, 'm0', m0);
ym = ymhat;
Nm = numel(ym);

A_sum = zeros(k,n); Sig_sum = zeros(n); h_sum = zeros(Te,1); phi_sum = 0; sigh2_sum = 0;
ym_sum = zeros(Nm,1);
store_ym = zeros(nsim, Nm, 'single');
max_resid = 0; n_accept = 0; n_mh = 0;
if keep_draws
    D = struct('A', zeros(nsim, k*n), 'Sig', zeros(nsim, n*n), 'h', zeros(nsim, Te), ...
        'phi', zeros(nsim,1), 'sigh2', zeros(nsim,1));
end

for isim = 1:nsim + burnin
        % ---- BLOCK 1: the missing values ----
    if isim > 1
        ym = bvar.samplers.missing_var(Y, A, Sig, h, p, 'M', M, 'z', z, 'V0', V0, 'm0', m0);
    end
    ycomp = Sel.So*Sel.yo + Sel.Sm*ym;
    [Yt, X] = bvar.util.build_lags(reshape(ycomp, n, T)', p);

        % ---- BLOCK 2: the coefficients, jointly ----
    w = exp(-h);
    A = bvar.samplers.var_coef_joint(Yt, X, Sig, w, iVb);

        % ---- BLOCK 3: Sig ----
    E = Yt - X*A;
    Shat = S0 + E'*(E.*w);
    Sig = iwishrnd((Shat + Shat')/2, nu0 + Te);

        % ---- BLOCK 4: h ----
    s2 = sum((E/chol(Sig, 'lower')').^2, 2);
    [h, na, nb] = bvar.sv.csv_armh_block(s2, phi, sigh2, h, n, 'block', hblock, ...
        'c_reject', c_reject, 'ForcedAccept', isim <= 20);
    if isim > 20
        n_accept = n_accept + na;  n_mh = n_mh + nb;
    end

        % ---- BLOCK 5: phi and sigh2 ----
    [phi, sigh2] = bvar.sv.sv0_params(h, phi, Hyper, 1, 'proposal', 'truncated');

    if isim > burnin
        i = isim - burnin;
        store_ym(i,:) = ym';  ym_sum = ym_sum + ym;
        A_sum = A_sum + A;  Sig_sum = Sig_sum + Sig;  h_sum = h_sum + h;
        phi_sum = phi_sum + phi;  sigh2_sum = sigh2_sum + sigh2;
        if ~isempty(M)
            max_resid = max(max_resid, norm(M*ycomp - z(:), inf));
        end
        if keep_draws
            D.A(i,:) = A(:)';  D.Sig(i,:) = Sig(:)';  D.h(i,:) = h';
            D.phi(i) = phi;  D.sigh2(i) = sigh2;
        end
    end
end

ymean = ym_sum/nsim;
res.Y_mean = reshape(Sel.So*Sel.yo + Sel.Sm*ymean, n, T)';
res.Y_q = repmat(res.Y_mean, 1, 1, numel(probs));
if Nm > 0
    qv = quantile(double(store_ym), probs, 1);     % numel(probs) x Nm
    for j = 1:numel(probs)
        res.Y_q(:,:,j) = reshape(Sel.So*Sel.yo + Sel.Sm*qv(j,:)', n, T)';
    end
end
res.A_mean = A_sum/nsim;  res.Sig_mean = Sig_sum/nsim;  res.h_mean = h_sum/nsim;
res.phi_mean = phi_sum/nsim;  res.sigh2_mean = sigh2_sum/nsim;
res.accept_rate = n_accept/max(n_mh, 1);
res.max_resid = max_resid;
res.sig2 = sig2;
res.quantiles = probs;
res.prior = struct('kappa', kappa, 'V', V, 'nu0', nu0, 'S0', S0, 'Hyper', Hyper, 'V0', V0, 'm0', m0);
res.nsim = nsim; res.burnin = burnin; res.seed = seed; res.p = p; res.c_reject = c_reject;
res.hblock = hblock;
if keep_draws
    D.ym = store_ym;
    res.draws = D;
end
end

function s2 = ar4_resid_var(y)
% residual variance of an AR(4) with intercept, over the periods whose value and four
% lags are all observed; NaN when fewer than 20 such periods exist
T = numel(y);
Z = [ones(T-4,1) y(4:T-1) y(3:T-2) y(2:T-3) y(1:T-4)];
yy = y(5:T);
ok = all(~isnan([Z yy]), 2);
if sum(ok) < 20
    s2 = NaN;
    return
end
b = Z(ok,:)\yy(ok);
s2 = mean((yy(ok) - Z(ok,:)*b).^2);
end
