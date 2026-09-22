% chan_eisenstat2018_jae_mltvpsv/run_all - functionized estimation pipeline of Chan and
% Eisenstat (2018, JAE): the ten structural VARs whose marginal likelihoods and DICs the
% paper compares, from the TVP-VAR with stochastic volatility to regime-switching VARs.
%
%   out = run_all(model, p, r, nsims, burnin, seed)
%   out = run_all(..., 'data', D)
%
%   model  - 'TVP-SV' | 'TVP' | 'TVP-R1-SV' | 'TVP-R2-SV' | 'TVP-R3-SV' | 'CVAR-SV' |
%            'CVAR' | 'RS-VAR' | 'RS-VAR-R1' | 'RS-VAR-R2', the paper's labels, or the
%            legacy code 1-10 (main_tvpsv.m lines 17-18, which call models 6-10
%            VAR-SV, VAR, RS-VAR, RS-VAR-1 and RS-VAR-2; these spellings are
%            accepted too); default 'CVAR-SV' (legacy default model = 6)
%   p      - number of lags, 1-4 (line 23); default 2
%   r      - number of regimes of the three RS models, at least 2 (line 22);
%            default 2; the other models ignore it
%   nsims, burnin - defaults 20000 / 5000 (lines 25-26)
%   seed   - when nonempty, rng(seed,'twister') is set before any draw;
%            omitted/empty uses the ambient stream as-is
%   'data' - a data matrix used in place of the package's file; its first 4 rows
%            are initial conditions, as with the file
%
% out holds the stored draws and the script-tail summaries under their legacy names,
% the design matrices and the log prior density (out.prior, a function handle) that
% the marginal-likelihood and DIC routines take, and Y (stacked T*n x 1), Y0 and
% shortY (T x n).
%
% Reproduces main_tvpsv.m and the model scripts draw for draw
% (tests/unit/test_mltvpsv_equivalence.m), with three deliberate divergences. The
% scripts re-seed the global stream from the wall clock and switch MATLAB to the v4/v5
% generators (TVPSV.m 53, TVP.m 48, TVP_R1_SV.m 61, TVP_R2_SV.m 71, TVP_R3_SV.m 76,
% VAR_SV.m 46, VAR.m 41, VAR_RS.m 63, VAR_RS_R1.m 62, VAR_RS_R2.m 64); run_all drops
% those lines so the caller controls seeding. It factors each non-diagonal precision
% matrix once and solves with that factor, where the scripts solve with backslash and
% factor again for the draw; the solutions differ in the last bits, so the test gives
% the legacy copies the substitutions of tests/unit/private/one_factor_patch.m. The
% wall-clock timing displays are not reproduced.
%
% Two legacy quirks are reproduced as published. In TVP-R3-SV the draw of mu_0 uses the
% current mu_0 as its prior mean (TVP_R3_SV.m 100), where the script defines the prior
% mean amu = 0 (line 15). In RS-VAR-R2 a regime with no observations redraws the common
% error variances from their prior inside the regime loop (VAR_RS_R2.m 77), so the
% coefficient draws of the later regimes in that sweep use the prior draw.
%
% Scope: estimation only. The marginal likelihoods (ml_*.m, intlike_*.m) and the DIC
% (dic_*.m) are separate phases.
%
% Core used: bvar.util.build_lags, bvar.util.surform (SURform.m), bvar.util.surform2
% (SURform2.m), bvar.sv.ksc_rw_h0 (SVRW.m). constructX.m, dirirnd.m and ldiripdf.m
% have no core counterpart and are copied below as local functions.
%
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for time-varying
% parameter VARs with stochastic volatility, Journal of Applied Econometrics, 33(4),
% 509-532.

function out = run_all(model, p, r, nsims, burnin, seed, varargin)
thisdir = fileparts(mfilename('fullpath'));

    % make bvar.* resolvable when called standalone
if isempty(which('bvar.sv.ksc_rw_h0'))
    addpath(fullfile(fileparts(fileparts(thisdir)), 'core'));
end

    % constants: preset.m in this folder (cd guard pins name resolution)
od = cd(thisdir);
guard = onCleanup(@() cd(od));
pr = preset();
clear guard

names = {'TVP-SV', 'TVP', 'TVP-R1-SV', 'TVP-R2-SV', 'TVP-R3-SV', ...
    'CVAR-SV', 'CVAR', 'RS-VAR', 'RS-VAR-R1', 'RS-VAR-R2'};
legacy_names = {'', '', '', '', '', 'VAR-SV', 'VAR', '', 'RS-VAR-1', 'RS-VAR-2'};
if nargin < 1 || isempty(model), model = pr.model_default; end
if isnumeric(model)
    assert(isscalar(model) && any(model == 1:10), 'run_all:model', 'numeric model must be 1-10');
    imodel = model;
else
    imodel = find(strcmpi(model, names), 1);
    if isempty(imodel)
        imodel = find(strcmpi(model, legacy_names), 1);
    end
    if isempty(imodel)
        error('run_all:model', 'model must be one of %s (or 1-10)', strjoin(names, ', '));
    end
end
if nargin < 2 || isempty(p),      p = pr.p_default;           end
if nargin < 3 || isempty(r),      r = pr.r_default;           end
if nargin < 4 || isempty(nsims),  nsims = pr.nsims_default;   end
if nargin < 5 || isempty(burnin), burnin = pr.burnin_default; end
if nargin < 6, seed = []; end
data_user = [];
if mod(numel(varargin), 2) ~= 0
    error('run_all:badOption', 'options must come in name-value pairs');
end
for iv = 1:2:numel(varargin)
    switch lower(char(string(varargin{iv})))
        case 'data', data_user = varargin{iv+1};
        otherwise, error('run_all:badOption', 'unknown option ''%s''', char(string(varargin{iv})));
    end
end
if ~(isnumeric(p) && isscalar(p) && p >= 1 && p <= pr.n0 && p == fix(p))
    error('run_all:badOption', 'p must be an integer from 1 to %d', pr.n0);
end
if ~(isnumeric(r) && isscalar(r) && r == fix(r) && r >= 1) || (imodel >= 8 && r < 2)
    error('run_all:badOption', 'r must be a positive integer, and at least 2 for the RS models');
end
if ~(isnumeric(nsims) && isscalar(nsims) && nsims >= 1 && nsims == fix(nsims)) || ...
        ~(isnumeric(burnin) && isscalar(burnin) && burnin >= 0 && burnin == fix(burnin))
    error('run_all:badOption', 'nsims must be a positive integer and burnin a nonnegative one');
end
if ~isempty(data_user) && ~(isnumeric(data_user) && ismatrix(data_user) ...
        && size(data_user,2) >= 2 && size(data_user,1) > pr.n0 + 1 && all(isfinite(data_user(:))))
    error('run_all:badData', ['data must be a finite matrix with at least 2 columns ' ...
        'and more than %d rows'], pr.n0 + 1);
end
if ~isempty(seed)
    rng(seed, 'twister');
end

    % data [main_tvpsv.m 35-41]
if isempty(data_user)
    USdata = xlsread(fullfile(thisdir, 'legacy', pr.data_file), pr.data_range);   % legacy folder, read-only
    data = USdata(:, pr.cols);
else
    data = double(data_user);
end
Y0 = data(1:pr.n0, :);
shortY = data(pr.n0+1:end, :);
[T, n] = size(shortY);
Y = reshape(shortY', T*n, 1);
    % every script builds the lag matrix X with the same inline loop; build_lags
    % returns [ones(T,1) X] with the lags copied, so its columns 2:end are X exactly
[~, Z1] = bvar.util.build_lags([Y0(end-p+1:end, :); shortY], p);
X = Z1(:, 2:end);

switch imodel
    case 1,  res = est_tvpsv(Y, shortY, X, T, n, p, nsims, burnin, pr);
    case 2,  res = est_tvp(Y, shortY, X, T, n, p, nsims, burnin, pr);
    case 3,  res = est_tvp_r1_sv(Y, shortY, X, T, n, p, nsims, burnin, pr);
    case 4,  res = est_tvp_r2_sv(Y, shortY, X, T, n, p, nsims, burnin, pr);
    case 5,  res = est_tvp_r3_sv(Y, shortY, X, T, n, p, nsims, burnin, pr);
    case 6,  res = est_cvarsv(Y, shortY, X, T, n, p, nsims, burnin, pr);
    case 7,  res = est_cvar(Y, shortY, X, T, n, p, nsims, burnin, pr);
    case 8,  res = est_rs(Y, shortY, X, T, n, p, r, nsims, burnin, pr);
    case 9,  res = est_rs_r1(Y, shortY, X, T, n, p, r, nsims, burnin, pr);
    case 10, res = est_rs_r2(Y, shortY, X, T, n, p, r, nsims, burnin, pr);
end

out = res;
out.model = names{imodel};
out.model_num = imodel;
out.p = p; out.r = r;
out.nsims = nsims; out.burnin = burnin; out.seed = seed;
out.is_package_data = isempty(data_user);
out.T = T; out.n = n;
out.Y = Y; out.Y0 = Y0; out.shortY = shortY;
out.preset = pr;
end

% -------------------------------------------------------------------------
function res = est_tvpsv(Y, shortY, X, T, n, p, nsims, burnin, pr)
% TVPSV.m functionized line-for-line. Draw order per sweep: theta -> theta0 -> h ->
% h0 -> Sigtheta -> Sigh.
m = n*(n-1)/2;      % dimension of the impact matrix [8]
k = n^2*p + n + m;  % dimension of states [9]

    % prior [10-22]
atheta = pr.prior_mean*ones(k,1); Vtheta = pr.prior_var*ones(k,1);
ah = pr.prior_mean*ones(n,1); Vh = pr.prior_var*ones(n,1);
nutheta0 = pr.nu*ones(k,1);
Stheta0 = pr.S_state*ones(k,1).*(nutheta0-1);
Stheta0(1:n*p+1:(k-n*(n-1)/2)) = pr.S_intercept*(nutheta0(1:n*p+1:(k-n*(n-1)/2))-1);
nuh0 = pr.nu*ones(n,1); Sh0 = pr.S_h*ones(n,1).*(nuh0-1);
cpri = nutheta0'*log(Stheta0) - sum(gammaln(nutheta0))...
    + nuh0'*log(Sh0) - sum(gammaln(nuh0)) ...
    -.5*(k+n)*log(2*pi) - .5*sum(log(Vtheta)) - .5*sum(log(Vh));
prior = @(sthe,sh,a0,b0) -(nutheta0+1)'*log(sthe) - sum(Stheta0./sthe) ...
    - (nuh0+1)'*log(sh) - sum(Sh0./sh) ...
    + cpri -.5*((a0-atheta)./Vtheta)'*(a0-atheta) -.5*((b0-ah)./Vh)'*(b0-ah);

    % design [24-37]
X2 = zeros(T,n*(n-1)/2);
count = 0;
for i=2:n
    X2(:,count+1:count+i-1) = -shortY(:,1:i-1);
    count = count + i-1;
end
bigX = construct_x(X,X2,n);
Htheta = speye(T*k) - sparse(k+1:T*k,1:(T-1)*k,ones((T-1)*k,1),T*k,T*k);

    % initialize [39-50]
store_Sigtheta = zeros(nsims,k);
store_Sigh = zeros(nsims,n);
store_theta = zeros(nsims,T*k);
store_theta0 = zeros(nsims,k);
store_h = zeros(nsims,T*n);
store_h0 = zeros(nsims,n);
Sigtheta = pr.init.state_var*ones(k,1);
Sigh = pr.init.sigh*ones(n,1);
h0 = log(var(shortY))';
h = repmat(h0',T,1);
theta0 = zeros(k,1);

disp('Starting TVP-SV.... ');   % [54]
for isim = 1:nsims+burnin
        % sample theta [59-68]
    invSig = sparse(1:T*n,1:T*n,reshape(exp(-h)',T*n,1));
    invS = sparse(1:T*k,1:T*k,repmat(1./Sigtheta',1,T),T*k,T*k);
    XinvSig = bigX'*invSig;
    HinvSH = Htheta'*invS*Htheta;
    alptheta = Htheta\[theta0;sparse((T-1)*k,1)];
    Ktheta = HinvSH + XinvSig*bigX;
    dtheta = XinvSig*Y + HinvSH*alptheta;
    CKtheta = chol(Ktheta,'lower');
    thetahat = CKtheta'\(CKtheta\dtheta);
    theta = thetahat + CKtheta'\randn(T*k,1);

        % sample theta0 [70-73]
    Ktheta0 = sparse(1:k,1:k,1./Sigtheta + 1./Vtheta);
    theta0hat = Ktheta0\(atheta./Vtheta + theta(1:k)./Sigtheta);
    theta0 = theta0hat + chol(Ktheta0,'lower')'\randn(k,1);

        % sample h [75-81]
    u = Y-bigX*theta;
    shortu = reshape(u,n,T)';
    for i=1:n
        Ystar = log(shortu(:,i).^2 + pr.sv_offset);
        h(:,i) = bvar.sv.ksc_rw_h0(Ystar,h(:,i),Sigh(i),h0(i));
    end

        % sample h0 [83-86]
    Kh0 = sparse(1:n,1:n,1./Sigh + 1./Vh);
    h0hat = Kh0\(ah./Vh + h(1,:)'./Sigh);
    h0 = h0hat + chol(Kh0,'lower')'\randn(n,1);

        % sample Sigtheta [88-90]
    e = reshape(theta-[theta0;theta(1:(T-1)*k)],k,T);
    Sigtheta = 1./gamrnd(nutheta0+T/2, 1./(Stheta0 + sum(e.^2,2)/2));

        % sample Sigh [92-94]
    e = h - [h0';h(1:T-1,:)];
    Sigh = 1./gamrnd(nuh0+T/2, 1./(Sh0 + sum(e.^2)'/2));

    if isim>burnin                                    % [96-104]
        i = isim-burnin;
        store_theta(i,:) = theta';
        store_h(i,:) = reshape(h',1,T*n);
        store_Sigtheta(i,:) = Sigtheta';
        store_Sigh(i,:) = Sigh';
        store_theta0(i,:) = theta0';
        store_h0(i,:) = h0';
    end

    if ( mod( isim, pr.progress_every ) ==0 )           % [106-108]
        disp(  [ num2str( isim ) ' loops... ' ] )
    end
end

res = struct('k',k, 'm',m, 'bigX',bigX, 'prior',prior, ...
    'store_theta',store_theta, 'store_h',store_h, 'store_Sigtheta',store_Sigtheta, ...
    'store_Sigh',store_Sigh, 'store_theta0',store_theta0, 'store_h0',store_h0);
end

% -------------------------------------------------------------------------
function res = est_tvp(Y, shortY, X, T, n, p, nsims, burnin, pr)
% TVP.m functionized line-for-line. Draw order per sweep: theta -> theta0 -> Sig ->
% Sigtheta.
m = n*(n-1)/2;      % dimension of the impact matrix [8]
k = n^2*p + n + m;  % dimension of states [9]

    % prior [11-21]
atheta = pr.prior_mean*ones(k,1); Vtheta = pr.prior_var*ones(k,1);
nutheta0 = pr.nu*ones(k,1);
Stheta0 = pr.S_state*ones(k,1).*(nutheta0-1);
Stheta0(1:n*p+1:(k-n*(n-1)/2)) = pr.S_intercept*(nutheta0(1:n*p+1:(k-n*(n-1)/2))-1);
nu0 = pr.nu*ones(n,1); S0 = pr.S_sig*ones(n,1).*(nu0-1);

cpri = nutheta0'*log(Stheta0) - sum(gammaln(nutheta0)) ...
    + nu0'*log(S0) - sum(gammaln(nu0)) - .5*k*log(2*pi) - .5*sum(log(Vtheta));
prior = @(s,sthe,a0) cpri -(nutheta0+1)'*log(sthe) - sum(Stheta0./sthe) ...
    - (nu0+1)'*log(s) - sum(S0./s) -.5*((a0-atheta)./Vtheta)'*(a0-atheta);

    % design [23-36]
X2 = zeros(T,n*(n-1)/2);
count = 0;
for i=2:n
    X2(:,count+1:count+i-1) = -shortY(:,1:i-1);
    count = count + i-1;
end
bigX = construct_x(X,X2,n);
Htheta = speye(T*k) - sparse(k+1:T*k,1:(T-1)*k,ones((T-1)*k,1),T*k,T*k);

    % initialize [38-45]
store_Sigtheta = zeros(nsims,k);
store_Sig = zeros(nsims,n);
store_theta = zeros(nsims,T*k);
store_theta0 = zeros(nsims,k);
Sigtheta = pr.init.state_var*ones(k,1);
Sig = pr.init.sig*ones(n,1);
theta0 = zeros(k,1);

disp('Starting TVP.... ');   % [49]
for isim = 1:nsims + burnin
        % sample theta [54-62]
    invS = sparse(1:T*k,1:T*k,repmat(1./Sigtheta',1,T),T*k,T*k);
    XinvSig = bigX'*sparse(1:T*n,1:T*n,repmat(1./Sig,T,1));
    HinvSH = Htheta'*invS*Htheta;
    alptheta = Htheta\[theta0;sparse((T-1)*k,1)];
    Ktheta = HinvSH + XinvSig*bigX;
    dtheta = XinvSig*Y + HinvSH*alptheta;
    CKtheta = chol(Ktheta,'lower');
    thetahat = CKtheta'\(CKtheta\dtheta);
    theta = thetahat + CKtheta'\randn(T*k,1);

        % sample theta0 [64-67]
    Ktheta0 = sparse(1:k,1:k,1./Sigtheta + 1./Vtheta);
    theta0hat = Ktheta0\(atheta./Vtheta + theta(1:k)./Sigtheta);
    theta0 = theta0hat + chol(Ktheta0,'lower')'\randn(k,1);

        % sample Sig [69-71]
    e = reshape(Y - bigX*theta,n,T)';
    Sig = 1./gamrnd(nu0+T/2, 1./(S0 + sum(e.^2)'/2));

        % sample Sigtheta [73-75]
    e = reshape(theta-[theta0;theta(1:(T-1)*k)],k,T);
    Sigtheta = 1./gamrnd(nutheta0+T/2, 1./(Stheta0 + sum(e.^2,2)/2));

    if isim>burnin                                    % [77-83]
        i = isim-burnin;
        store_theta(i,:) = theta';
        store_Sigtheta(i,:) = Sigtheta';
        store_Sig(i,:) = Sig';
        store_theta0(i,:) = theta0';
    end

    if ( mod( isim, pr.progress_every ) ==0 )           % [85-87]
        disp(  [ num2str( isim ) ' loops... ' ] )
    end
end

res = struct('k',k, 'm',m, 'bigX',bigX, 'prior',prior, ...
    'store_theta',store_theta, 'store_Sigtheta',store_Sigtheta, ...
    'store_Sig',store_Sig, 'store_theta0',store_theta0);
end

% -------------------------------------------------------------------------
function res = est_tvp_r1_sv(Y, shortY, X, T, n, p, nsims, burnin, pr)
% TVP_R1_SV.m functionized line-for-line: constant VAR coefficients beta, time-varying
% impact elements gam_t. Draw order per sweep: gam -> beta -> gam0 -> h -> h0 ->
% Siggam -> Sigh.
kgam = n*(n-1)/2;       % [8]
kbeta = n^2*p + n;      % [9]

    % prior [11-25]; beta0 here is the prior mean of beta
beta0 = pr.prior_mean*ones(kbeta,1); Vbeta = pr.prior_var*ones(kbeta,1);
agam = pr.prior_mean*ones(kgam,1); Vgam = pr.prior_var*ones(kgam,1);
ah = pr.prior_mean*ones(n,1); Vh = pr.prior_var*ones(n,1);
nugam0 = pr.nu*ones(kgam,1); Sgam0 = pr.S_state*ones(kgam,1).*(nugam0-1);
nuh0 = pr.nu*ones(n,1); Sh0 = pr.S_h*ones(n,1).*(nuh0-1);

cpri = -.5*kbeta*log(2*pi) - .5*sum(log(Vbeta)) ...
    + nugam0'*log(Sgam0) - sum(gammaln(nugam0))...
    + nuh0'*log(Sh0) - sum(gammaln(nuh0)) ...
    -.5*(kgam+n)*log(2*pi) - .5*sum(log(Vgam)) - .5*sum(log(Vh));
prior = @(b,sg,sh,a0,b0) cpri -.5*((b-beta0)./Vbeta)'*(b-beta0) ...
    - (nugam0+1)'*log(sg) - sum(Sgam0./sg) ...
    - (nuh0+1)'*log(sh) - sum(Sh0./sh) ...
    - .5*((a0-agam)./Vgam)'*(a0-agam) -.5*((b0-ah)./Vh)'*(b0-ah);

    % design [27-41]
Xtilde = bvar.util.surform2([ones(T,1) X],n);
X2 = zeros(T,n*(n-1)/2);
count = 0;
for i=2:n
    X2(:,count+1:count+i-1) = -shortY(:,1:i-1);
    count = count + i-1;
end
W = construct_x([],X2,n);
Hgam = speye(T*kgam) - sparse(kgam+1:T*kgam,1:(T-1)*kgam,ones((T-1)*kgam,1),T*kgam,T*kgam);

    % initialize [43-57]
store_Siggam = zeros(nsims,kgam);
store_Sigh = zeros(nsims,n);
store_beta = zeros(nsims,kbeta);
store_gam = zeros(nsims,T*kgam);
store_gam0 = zeros(nsims,kgam);
store_h = zeros(nsims,T*n);
store_h0 = zeros(nsims,n);

beta = (Xtilde'*Xtilde)\(Xtilde'*Y);
Siggam = pr.init.state_var*ones(kgam,1);
Sigh = pr.init.sigh*ones(n,1);
h0 = log(var(shortY))';
h = repmat(h0',T,1);
gam0 = zeros(kgam,1);

disp('Starting TVP-R1-SV.... ');   % [62]
for isim = 1:nsims + burnin
        % sample gam [67-75]
    invSig = sparse(1:T*n,1:T*n,reshape(exp(-h)',T*n,1));
    invS = sparse(1:T*kgam,1:T*kgam,repmat(1./Siggam',1,T));
    WinvSig = W'*invSig;
    HinvSH = Hgam'*invS*Hgam;
    alpgam = Hgam\[gam0;sparse((T-1)*kgam,1)];
    Kgam = HinvSH + WinvSig*W;
    CKgam = chol(Kgam,'lower');
    gamhat = CKgam'\(CKgam\(HinvSH*alpgam + WinvSig*(Y-Xtilde*beta)));
    gam = gamhat + CKgam'\randn(T*kgam,1);

        % sample beta [77-81]
    XtildeinvSig = Xtilde'*invSig;
    Kbeta = sparse(1:kbeta,1:kbeta,1./Vbeta) + XtildeinvSig*Xtilde;
    CKbeta = chol(Kbeta,'lower');
    betahat = CKbeta'\(CKbeta\(beta0./Vbeta + XtildeinvSig*(Y-W*gam)));
    beta = betahat + CKbeta'\randn(kbeta,1);

        % sample gam0 [83-86]
    Kgam0 = sparse(1:kgam,1:kgam,1./Siggam + 1./Vgam);
    gam0hat = Kgam0\(agam./Vgam + gam(1:kgam)./Siggam);
    gam0 = gam0hat + chol(Kgam0,'lower')'\randn(kgam,1);

        % sample h [88-94]
    u = Y-Xtilde*beta - W*gam;
    shortu = reshape(u,n,T)';
    for i=1:n
        Ystar = log(shortu(:,i).^2 + pr.sv_offset );
        h(:,i) = bvar.sv.ksc_rw_h0(Ystar,h(:,i),Sigh(i),h0(i));
    end

        % sample h0 [96-99]
    Kh0 = sparse(1:n,1:n,1./Sigh + 1./Vh);
    h0hat = Kh0\(ah./Vh + h(1,:)'./Sigh);
    h0 = h0hat + chol(Kh0,'lower')'\randn(n,1);

        % sample Siggam [101-103]
    e = reshape(gam-[gam0;gam(1:(T-1)*kgam)],kgam,T);
    Siggam = 1./gamrnd(nugam0+T/2, 1./(Sgam0 + sum(e.^2,2)/2));

        % sample Sigh [105-107]
    e = h - [h0';h(1:T-1,:)];
    Sigh = 1./gamrnd(nuh0+T/2, 1./(Sh0 + sum(e.^2)'/2));

    if isim>burnin                                    % [109-118]
        i = isim-burnin;
        store_beta(i,:) =beta';
        store_gam(i,:) = gam';
        store_h(i,:) = reshape(h',1,T*n);
        store_Siggam(i,:) = Siggam';
        store_Sigh(i,:) = Sigh';
        store_gam0(i,:) = gam0';
        store_h0(i,:) = h0';
    end

    if ( mod( isim, pr.progress_every ) ==0 )           % [120-122]
        disp(  [ num2str( isim ) ' loops... ' ] )
    end
end

res = struct('kgam',kgam, 'kbeta',kbeta, 'Xtilde',Xtilde, 'W',W, 'prior',prior, ...
    'store_beta',store_beta, 'store_gam',store_gam, 'store_h',store_h, ...
    'store_Siggam',store_Siggam, 'store_Sigh',store_Sigh, 'store_gam0',store_gam0, ...
    'store_h0',store_h0);
    % posterior summaries [127-130]
res.betahat = mean(store_beta)';
res.betaCI = quantile(store_beta,[.05 .95])';
res.gamhat = mean(store_gam)';
res.gamCI = quantile(store_gam,[.05 .95])';
end

% -------------------------------------------------------------------------
function res = est_tvp_r2_sv(Y, shortY, X, T, n, p, nsims, burnin, pr)
% TVP_R2_SV.m functionized line-for-line: time-varying VAR coefficients beta_t,
% constant impact elements gam. Draw order per sweep: beta -> gam -> beta0 -> h -> h0
% -> Sigbeta -> Sigh.
kgam = n*(n-1)/2;       % [8]
kbeta = n^2*p + n;      % [9]

    % prior [11-27]; gam0 here is the prior mean of gam
gam0 = pr.prior_mean*ones(kgam,1); Vgam = pr.prior_var*ones(kgam,1);
abeta = pr.prior_mean*ones(kbeta,1); Vbeta = pr.prior_var*ones(kbeta,1);
ah = pr.prior_mean*ones(n,1); Vh = pr.prior_var*ones(n,1);
nubeta0 = pr.nu*ones(kbeta,1);
Sbeta0 = pr.S_state*ones(kbeta,1).*(nubeta0-1);
Sbeta0(1:n*p+1:end) = pr.S_intercept*(nubeta0(1:n*p+1:end)-1);
nuh0 = pr.nu*ones(n,1); Sh0 = pr.S_h*ones(n,1).*(nuh0-1);

cpri = -.5*kgam*log(2*pi) - .5*sum(log(Vgam)) ...
    + nubeta0'*log(Sbeta0) - sum(gammaln(nubeta0))...
    + nuh0'*log(Sh0) - sum(gammaln(nuh0)) ...
    -.5*(kbeta+n)*log(2*pi) - .5*sum(log(Vbeta)) - .5*sum(log(Vh));
prior = @(g,sb,sh,a0,b0) cpri -.5*((g-gam0)./Vgam)'*(g-gam0) ...
    - (nubeta0+1)'*log(sb) - sum(Sbeta0./sb) ...
    - (nuh0+1)'*log(sh) - sum(Sh0./sh) ...
    - .5*((a0-abeta)./Vbeta)'*(a0-abeta) -.5*((b0-ah)./Vh)'*(b0-ah);

    % design [29-52]
Xtilde = bvar.util.surform([ones(n*T,1) kron(X,ones(n,1))]);
X2 = zeros(T,n*(n-1)/2);
count = 0;
for i=2:n
    X2(:,count+1:count+i-1) = -shortY(:,1:i-1);
    count = count + i-1;
end
idj = repmat(1:kgam,1,T)';
idi = zeros(T*kgam,1);
count = 0;
for i=2:n
    for j=1:i-1
        idi(count+1:kgam:end) = i:n:T*n;
        count = count + 1;
    end
end
W = sparse(idi,idj,reshape(X2',T*kgam,1));
Hbeta = speye(T*kbeta) - sparse(kbeta+1:T*kbeta,1:(T-1)*kbeta,ones((T-1)*kbeta,1),T*kbeta,T*kbeta);

    % initialize [54-68]; beta0 here is the initial state of beta_t
store_Sigbeta = zeros(nsims,kbeta);
store_Sigh = zeros(nsims,n);
store_beta = zeros(nsims,T*kbeta);
store_gam = zeros(nsims,kgam);
store_beta0 = zeros(nsims,kbeta);
store_h = zeros(nsims,T*n);
store_h0 = zeros(nsims,n);

gam = (W'*W)\(W'*Y);
Sigbeta = pr.init.state_var*ones(kbeta,1);
Sigh = pr.init.sigh*ones(n,1);
h0 = log(var(shortY))';
h = repmat(h0',T,1);
beta0 = zeros(kbeta,1);

disp('Starting TVP-R2-SV.... ');   % [72]
for isim = 1:nsims + burnin
        % sample beta [77-85]
    invSig = sparse(1:T*n,1:T*n,reshape(exp(-h)',T*n,1));
    invS = sparse(1:T*kbeta,1:T*kbeta,repmat(1./Sigbeta',1,T));
    XinvSig = Xtilde'*invSig;
    HinvSH = Hbeta'*invS*Hbeta;
    alpbeta = Hbeta\[beta0;sparse((T-1)*kbeta,1)];
    Kbeta = HinvSH + XinvSig*Xtilde;
    CKbeta = chol(Kbeta,'lower');
    betahat = CKbeta'\(CKbeta\(HinvSH*alpbeta + XinvSig*(Y-W*gam)));
    beta = betahat + CKbeta'\randn(T*kbeta,1);

        % sample gam [87-91]
    WinvSig = W'*invSig;
    Kgam = sparse(1:kgam,1:kgam,1./Vgam) + WinvSig*W;
    CKgam = chol(Kgam,'lower');
    gamhat = CKgam'\(CKgam\(gam0./Vgam + WinvSig*(Y-Xtilde*beta)));
    gam = gamhat + CKgam'\randn(kgam,1);

        % sample beta0 [93-96]
    Kbeta0 = sparse(1:kbeta,1:kbeta,1./Sigbeta + 1./Vbeta);
    beta0hat = Kbeta0\(abeta./Vbeta + beta(1:kbeta)./Sigbeta);
    beta0 = beta0hat + chol(Kbeta0,'lower')'\randn(kbeta,1);

        % sample h [98-104]
    u = Y-Xtilde*beta-W*gam;
    shortu = reshape(u,n,T)';
    for i=1:n
        Ystar = log(shortu(:,i).^2 + pr.sv_offset );
        h(:,i) = bvar.sv.ksc_rw_h0(Ystar,h(:,i),Sigh(i),h0(i));
    end

        % sample h0 [106-109]
    Kh0 = sparse(1:n,1:n,1./Sigh + 1./Vh);
    h0hat = Kh0\(ah./Vh + h(1,:)'./Sigh);
    h0 = h0hat + chol(Kh0,'lower')'\randn(n,1);

        % sample Sigbeta [111-113]
    e = reshape(beta-[beta0;beta(1:(T-1)*kbeta)],kbeta,T);
    Sigbeta = 1./gamrnd(nubeta0+T/2, 1./(Sbeta0 + sum(e.^2,2)/2));

        % sample Sigh [115-117]
    e = h - [h0';h(1:T-1,:)];
    Sigh = 1./gamrnd(nuh0+T/2, 1./(Sh0 + sum(e.^2)'/2));

    if isim>burnin                                    % [119-128]
        i = isim-burnin;
        store_beta(i,:) =beta';
        store_gam(i,:) = gam';
        store_h(i,:) = reshape(h',1,T*n);
        store_Sigbeta(i,:) = Sigbeta';
        store_Sigh(i,:) = Sigh';
        store_beta0(i,:) = beta0';
        store_h0(i,:) = h0';
    end

    if ( mod( isim, pr.progress_every ) ==0 )           % [130-132]
        disp(  [ num2str( isim ) ' loops... ' ] )
    end
end

res = struct('kgam',kgam, 'kbeta',kbeta, 'Xtilde',Xtilde, 'W',W, 'prior',prior, ...
    'store_beta',store_beta, 'store_gam',store_gam, 'store_h',store_h, ...
    'store_Sigbeta',store_Sigbeta, 'store_Sigh',store_Sigh, 'store_beta0',store_beta0, ...
    'store_h0',store_h0);
    % posterior summaries [137-140]
res.betahat = mean(store_beta)';
res.betaCI = quantile(store_beta,[.05 .95])';
res.gamhat = mean(store_gam)';
res.gamCI = quantile(store_gam,[.05 .95])';
end

% -------------------------------------------------------------------------
function res = est_tvp_r3_sv(Y, shortY, X, T, n, p, nsims, burnin, pr)
% TVP_R3_SV.m functionized line-for-line: time-varying intercepts mu_t, constant
% slopes beta and impact elements gam. Draw order per sweep: mu -> (beta,gam) -> mu0
% -> h -> h0 -> Sigmu -> Sigh.
kgam = n*(n-1)/2;       % [8]
kbeta = n^2*p;          % [9]

    % prior [11-28]; beta0 and gam0 here are the prior means of beta and gam
beta0 = pr.prior_mean*ones(kbeta,1); Vbeta = pr.prior_var*ones(kbeta,1);
gam0 = pr.prior_mean*ones(kgam,1); Vgam = pr.prior_var*ones(kgam,1);
ah = pr.prior_mean*ones(n,1); Vh = pr.prior_var*ones(n,1);
amu = pr.prior_mean*ones(n,1); Vmu = pr.prior_var*ones(n,1);
numu0 = pr.nu*ones(n,1); Smu0 = pr.S_intercept*ones(n,1).*(numu0-1);
nuh0 = pr.nu*ones(n,1); Sh0 = pr.S_h*ones(n,1).*(nuh0-1);

cpri = -.5*kbeta*log(2*pi) - .5*sum(log(Vbeta)) ...
    -.5*kgam*log(2*pi) - .5*sum(log(Vgam)) ...
    + numu0'*log(Smu0) - sum(gammaln(numu0))...
    + nuh0'*log(Sh0) - sum(gammaln(nuh0)) ...
    -.5*(2*n)*log(2*pi) - .5*sum(log(Vmu)) - .5*sum(log(Vh));
prior = @(b,g,sm,sh,a0,b0) cpri -.5*((b-beta0)./Vbeta)'*(b-beta0) ...
    -.5*((g-gam0)./Vgam)'*(g-gam0) ...
    - (numu0+1)'*log(sm) - sum(Smu0./sm) ...
    - (nuh0+1)'*log(sh) - sum(Sh0./sh) ...
    - .5*((a0-amu)./Vmu)'*(a0-amu) -.5*((b0-ah)./Vh)'*(b0-ah);

    % design [30-54]
Ztilde = bvar.util.surform2(X,n);
X2 = zeros(T,n*(n-1)/2);
count = 0;
for i=2:n
    X2(:,count+1:count+i-1) = -shortY(:,1:i-1);
    count = count + i-1;
end
idj = repmat(1:kgam,1,T)';
idi = zeros(T*kgam,1);
count = 0;
for i=2:n
    for j=1:i-1
        idi(count+1:kgam:end) = i:n:T*n;
        count = count + 1;
    end
end
W = sparse(idi,idj,reshape(X2',T*kgam,1));
Z = [Ztilde W];
Hmu = speye(T*n) - sparse(n+1:T*n,1:(T-1)*n,ones((T-1)*n,1),T*n,T*n);

    % initialize [56-73]
store_Sigmu = zeros(nsims,n);
store_Sigh = zeros(nsims,n);
store_mu = zeros(nsims,T*n);
store_gam = zeros(nsims,kgam);
store_beta = zeros(nsims,kbeta);
store_h = zeros(nsims,T*n);
store_h0 = zeros(nsims,n);
store_mu0 = zeros(nsims,n);

mu0 = mean(shortY)';
mu = repmat(mu0,T,1);
beta = (Ztilde'*Ztilde)\(Ztilde'*(Y-mu));
gam = (W'*W)\(W'*(Y-mu-Ztilde*beta));
Sigmu = pr.init.sigmu_r3*ones(n,1);
Sigh = pr.init.sigh_r3*ones(n,1);
h0 = log(mean(reshape(Y - mu - Ztilde*beta - W*gam,n,T).^2,2));
h = repmat(h0',T,1);

disp('Starting TVP-R3-SV.... ');   % [77]
for isim = 1:nsims + burnin
        % sample mu [82-88]
    invSig = sparse(1:T*n,1:T*n,reshape(exp(-h)',T*n,1));
    HinvSH_mu = Hmu'*sparse(1:T*n,1:T*n,repmat(1./Sigmu',1,T))*Hmu;
    alpmu = Hmu\[mu0;sparse((T-1)*n,1)];
    Kmu = HinvSH_mu + invSig;
    CKmu = chol(Kmu,'lower');
    muhat = CKmu'\(CKmu\(HinvSH_mu*alpmu + invSig*(Y-Ztilde*beta-W*gam)));
    mu = muhat + CKmu'\randn(T*n,1);

        % sample beta and gam [90-96]
    ZinvSig = [Ztilde W]'*invSig;
    Kbeta = sparse(1:kbeta+kgam,1:kbeta+kgam,[1./Vbeta;1./Vgam]) + ZinvSig*[Ztilde W];
    CKbeta = chol(Kbeta,'lower');
    beta_hat = CKbeta'\(CKbeta\([beta0./Vbeta;gam0./Vgam] + ZinvSig*(Y-mu)));
    draw = beta_hat + CKbeta'\randn(kbeta+kgam,1);
    beta = draw(1:kbeta);
    gam = draw(kbeta+1:end);

        % sample mu0 [98-101]; the prior mean is the current mu0 (quirk, see header)
    Kmu0 = sparse(1:n,1:n,1./Sigmu + 1./Vmu);
    mu0hat = Kmu0\(mu0./Vmu + mu(1:n)./Sigmu);
    mu0 = mu0hat + chol(Kmu0,'lower')'\randn(n,1);

        % sample h [103-109]
    u = Y-mu-Ztilde*beta-W*gam;
    shortu = reshape(u,n,T)';
    for i=1:n
        Ystar = log(shortu(:,i).^2 + pr.sv_offset );
        h(:,i) = bvar.sv.ksc_rw_h0(Ystar,h(:,i),Sigh(i),h0(i));
    end

        % sample h0 [111-114]
    Kh0 = sparse(1:n,1:n,1./Sigh + 1./Vh);
    h0hat = Kh0\(ah./Vh + h(1,:)'./Sigh);
    h0 = h0hat + chol(Kh0,'lower')'\randn(n,1);

        % sample Sigmu [116-118]
    e = reshape(mu-[mu0;mu(1:(T-1)*n)],n,T);
    Sigmu = 1./gamrnd(numu0+T/2, 1./(Smu0 + sum(e.^2,2)/2));

        % sample Sigh [120-122]
    e = h - [h0';h(1:T-1,:)];
    Sigh = 1./gamrnd(nuh0+T/2, 1./(Sh0 + sum(e.^2)'/2));

    if isim>burnin                                    % [124-134]
        isave = isim-burnin;
        store_mu(isave,:) = mu';
        store_beta(isave,:) = beta';
        store_gam(isave,:) = gam';
        store_h(isave,:) = reshape(h',1,T*n);
        store_Sigmu(isave,:) = Sigmu';
        store_Sigh(isave,:) = Sigh';
        store_mu0(isave,:) = mu0';
        store_h0(isave,:) = h0';
    end

    if (mod(isim, pr.progress_every_r3) == 0)         % [136-138]
        disp([num2str(isim) ' loops... ' ])
    end
end
disp(' ' );   % [143]

res = struct('kgam',kgam, 'kbeta',kbeta, 'Ztilde',Ztilde, 'W',W, 'Z',Z, 'prior',prior, ...
    'store_mu',store_mu, 'store_beta',store_beta, 'store_gam',store_gam, ...
    'store_h',store_h, 'store_Sigmu',store_Sigmu, 'store_Sigh',store_Sigh, ...
    'store_mu0',store_mu0, 'store_h0',store_h0);
    % posterior summaries [145-149]
res.muhat = mean(store_mu)';
res.muCI = quantile(store_mu,[.05 .95])';
res.beta_hat = mean(store_beta)';
res.gam_hat = mean(store_gam)';
res.gamCI = quantile(store_gam,[.05 .95])';
end

% -------------------------------------------------------------------------
function res = est_cvarsv(Y, shortY, X, T, n, p, nsims, burnin, pr)
% VAR_SV.m functionized line-for-line: constant coefficients theta = (VAR
% coefficients, impact elements), random-walk log-volatilities. Draw order per sweep:
% theta -> h -> h0 -> Sigh.
m = n*(n-1)/2;      % dimension of the impact matrix [8]
k = n^2*p + n + m;  % dimension of states [9]

    % prior [11-19]
atheta = pr.prior_mean*ones(k,1); Vtheta = pr.prior_var*ones(k,1);
ah = pr.prior_mean*ones(n,1); Vh = pr.prior_var*ones(n,1);
nuh0 = pr.nu*ones(n,1); Sh0 = pr.S_h*ones(n,1).*(nuh0-1);

cpri = -.5*(n+k)*log(2*pi) -.5*sum(log(Vtheta)) -.5*sum(log(Vh)) ...
    + nuh0'*log(Sh0) - sum(gammaln(nuh0));
prior = @(the,sh,b0) cpri -.5*(the-atheta)'*((the-atheta)./Vtheta) ...
    -(nuh0+1)'*log(sh) - sum(Sh0./sh) -.5*((b0-ah)./Vh)'*(b0-ah);

    % design [21-34]
X2 = zeros(n*T,n*(n-1)/2);
count = 0;
for i=2:n
    X2(i:n:end,count+1:count+i-1) = -shortY(:,1:i-1);
    count = count + i-1;
end
X1 = bvar.util.surform2([ones(T,1) X],n);
bigX = [X1 sparse(X2)];

    % initialize [36-43]
store_Sigh = zeros(nsims + burnin - burnin,n);
store_theta = zeros(nsims + burnin - burnin,k);
store_h = zeros(nsims + burnin - burnin,T*n);
store_h0 = zeros(nsims + burnin - burnin,n);
Sigh = pr.init.sigh_cvarsv*ones(n,1);
h0 = log(var(shortY))';
h = repmat(h0',T,1);

disp('Starting VAR-SV.... ');   % [47]
for isim = 1:nsims + burnin
        % sample theta [52-57]
    invSig = sparse(1:T*n,1:T*n,reshape(exp(-h)',T*n,1));
    XinvSig = bigX'*invSig;
    Ktheta = sparse(1:k,1:k,1./Vtheta) + XinvSig*bigX;
    CKtheta = chol(Ktheta,'lower');
    thetahat = CKtheta'\(CKtheta\(atheta./Vtheta + XinvSig*Y));
    theta = thetahat + CKtheta'\randn(k,1);

        % sample h [59-64]
    shortu = reshape(Y-bigX*theta,n,T)';
    for i=1:n
        Ystar = log(shortu(:,i).^2 + pr.sv_offset );
        h(:,i) = bvar.sv.ksc_rw_h0(Ystar,h(:,i),Sigh(i),h0(i));
    end

        % sample h0 [66-69]
    Kh0 = sparse(1:n,1:n,1./Sigh + 1./Vh);
    h0hat = Kh0\(ah./Vh + h(1,:)'./Sigh);
    h0 = h0hat + chol(Kh0,'lower')'\randn(n,1);

        % sample Sigh [71-73]
    e = h - [h0';h(1:T-1,:)];
    Sigh = 1./gamrnd(nuh0+T/2, 1./(Sh0 + sum(e.^2)'/2));

    if isim>burnin                                    % [75-81]
        i = isim-burnin;
        store_h(i,:) = reshape(h',1,T*n);
        store_theta(i,:) = theta';
        store_Sigh(i,:) = Sigh';
        store_h0(i,:) = h0';
    end

    if ( mod( isim, pr.progress_every ) ==0 )           % [83-85]
        disp(  [ num2str( isim ) ' loops... ' ] )
    end
end

res = struct('k',k, 'm',m, 'bigX',bigX, 'prior',prior, ...
    'store_theta',store_theta, 'store_h',store_h, 'store_Sigh',store_Sigh, ...
    'store_h0',store_h0);
    % posterior summaries [91-94]
res.thetahat = mean(store_theta)';
res.thetaCI = quantile(store_theta,[.05 .95])';
res.hhat = mean(store_h)';
res.hCI = quantile(store_h,[.05 .95])';
end

% -------------------------------------------------------------------------
function res = est_cvar(Y, shortY, X, T, n, p, nsims, burnin, pr)
% VAR.m functionized line-for-line: constant coefficients and homoskedastic errors.
% Draw order per sweep: theta -> Sig.
m = n*(n-1)/2;      % dimension of the impact matrix [8]
k = n^2*p + n + m;  % dimension of states [9]

    % prior [11-17]
atheta = pr.prior_mean*ones(k,1); Vtheta = pr.prior_var*ones(k,1);
nu0 = pr.nu*ones(n,1); S0 = pr.S_sig*ones(n,1).*(nu0-1);

cpri = -.5*k*log(2*pi) - .5*sum(log(Vtheta)) + nu0'*log(S0) - sum(gammaln(nu0));
prior = @(the,s) cpri -.5*(the-atheta)'*((the-atheta)./Vtheta) ...
    -(nu0+1)'*log(s) - sum(S0./s);

    % design [19-32]
X2 = zeros(n*T,n*(n-1)/2);
count = 0;
for i=2:n
    X2(i:n:end,count+1:count+i-1) = -shortY(:,1:i-1);
    count = count + i-1;
end
X1 = bvar.util.surform2([ones(T,1) X],n);
bigX = [X1 sparse(X2)];

    % initialize [35-38]
store_Sig = zeros(nsims,n);
store_theta = zeros(nsims,k);
Sig = pr.init.sig*ones(n,1);

disp('Starting VAR.... ');   % [42]
for isim = 1:nsims + burnin
        % sample theta [47-51]
    XinvSig = bigX'*sparse(1:T*n,1:T*n,repmat(1./Sig,T,1));
    Ktheta = sparse(1:k,1:k,1./Vtheta) + XinvSig*bigX;
    CKtheta = chol(Ktheta,'lower');
    theta_hat = CKtheta'\(CKtheta\(atheta./Vtheta + XinvSig*Y));
    theta = theta_hat + CKtheta'\randn(k,1);

        % sample Sig [53-55]
    e = reshape(Y - bigX*theta,n,T)';
    Sig = 1./gamrnd(nu0+T/2, 1./(S0 + sum(e.^2)'/2));

    if isim>burnin                                    % [57-61]
        i = isim-burnin;
        store_theta(i,:) = theta';
        store_Sig(i,:) = Sig';
    end

    if (mod(isim, pr.progress_every) == 0)            % [63-65]
        disp([num2str(isim) ' loops... '])
    end
end

res = struct('k',k, 'm',m, 'bigX',bigX, 'prior',prior, ...
    'store_theta',store_theta, 'store_Sig',store_Sig);
    % posterior summaries [70-71]
res.theta_hat = mean(store_theta)';
res.thetaCI = quantile(store_theta,[.05 .95])';
end

% -------------------------------------------------------------------------
function res = est_rs(Y, shortY, X, T, n, p, r, nsims, burnin, pr)
% VAR_RS.m functionized line-for-line: every coefficient and error variance switches
% with a first-order Markov chain of r regimes. Draw order per sweep: (theta_i, Sig_i)
% for each regime -> S by forward filtering, backward sampling -> the rows of P.
m = n*(n-1)/2;      % dimension of the impact matrix [8]
k = n^2*p + n + m;  % dimension of states [9]

    % prior [11-18]; the handle's third argument is the transition matrix
atheta = pr.prior_mean*ones(k,1); Vtheta = pr.prior_var*ones(k,1);
nu0 = pr.nu*ones(n,1); S0 = pr.S_sig*ones(n,1).*(nu0-1);
alp0 = pr.rs.alp0*ones(r,1);  % symmetric prior

cpri = -.5*k*r*log(2*pi) - .5*r*sum(log(Vtheta)) + r*nu0'*log(S0) - r*sum(gammaln(nu0));
prior = @(the,s,Pm) cpri -.5*(the-repmat(atheta,r,1))'*((the-repmat(atheta,r,1))./repmat(Vtheta,r,1)) ...
    -(repmat(nu0,r,1)+1)'*log(s) - sum(repmat(S0,r,1)./s) + sum(ldiripdf(Pm,alp0)) ;

    % design [20-32]
X2 = zeros(n*T,n*(n-1)/2);
count = 0;
for j=2:n
    X2(j:n:end,count+1:count+j-1) = -shortY(:,1:j-1);
    count = count + j-1;
end
bigX = [bvar.util.surform2([ones(T,1) X],n) sparse(X2)];

    % initialize the Markov chain [34-50]
S = [kron((1:r-1)',ones(floor(T/r),1));r*ones(T-floor(T/r)*(r-1),1)];
P = triu(pr.rs.P_move*ones(r,r)/(r-1),1) + tril(pr.rs.P_move*ones(r,r)/(r-1),-1);
P = P + pr.rs.P_stay*eye(r);

theta = zeros(k,r);
Sig = zeros(n,r);
for i=1:r
    idx = (S == i);
    Ti = sum(idx);
    Z = [ones(Ti,1) X(idx,:)];
    shortYi = shortY(idx,:);
    tmptheta = (Z'*Z)\(Z'*shortYi);
    E = shortYi - Z*tmptheta;
    theta(1:n^2*p+n,i) = reshape(tmptheta',n^2*p+n,1);
    Sig(:,i) = sum(E.^2)'/Ti;
end

    % initialize for storage [52-59]
store_Sig = zeros(nsims,n*r);
store_theta = zeros(nsims,k*r);
store_P = zeros(nsims,r,r);
store_S = zeros(T,r);
like = zeros(T,r);
tmpP1 = zeros(T,r);                   % p(s_t|Y_t,\theta,P)
tmpP2 = zeros(T,r); tmpP2(1,:) = pr.rs.p2_first; % p(s_t|Y_{t-1},\theta,P)

disp(['Starting VAR-RS with ' num2str(r) ' regimes.... ']);   % [64]
count_empty = 0;
for isim = 1:nsims + burnin
    for i=1:r
            % extract data for state S_t == i [69-71]
        idx = (S == i);
        Ti = sum(idx);
        if Ti == 0 % not an active regime, draw from prior [72-74]
            count_empty = count_empty + 1;
            theta(:,i) = atheta + sqrt(Vtheta).*randn(k,1);
            Sig(:,i) = 1./gamrnd(nu0,1./S0);
        else
            shortYi = shortY(idx,:);   %#ok<NASGU> % dead in the legacy too [76]
            Yi = reshape(shortY(idx,:)',Ti*n,1);

            X2i = zeros(n*Ti,n*(n-1)/2);
            count = 0;
            for j=2:n
                X2i(j:n:end,count+1:count+j-1) = -shortY(idx,1:j-1);
                count = count + j-1;
            end
            bigXi = [bvar.util.surform2([ones(Ti,1) X(idx,:)],n) sparse(X2i)];

                % sample theta [87-91]
            XiSig = bigXi'*sparse(1:Ti*n,1:Ti*n,repmat(1./Sig(:,i),Ti,1));
            Ktheta = sparse(1:k,1:k,1./Vtheta) + XiSig*bigXi;
            CKtheta = chol(Ktheta,'lower');
            theta_hat = CKtheta'\(CKtheta\(atheta./Vtheta + XiSig*Yi));
            theta(:,i) = theta_hat + CKtheta'\randn(k,1);

                % sample Sig [93-95]
            e = reshape(Yi - bigXi*theta(:,i),n,Ti)';
            Sig(:,i) = 1./gamrnd(nu0+Ti/2, 1./(S0 + sum(e.^2)'/2));
        end
    end
        % sample S [98-119]
    for i=1:r
        mu_i = bigX*theta(:,i);
        Sig_i = Sig(:,i);
        like(:,i) = mvnpdf(shortY,reshape(mu_i,n,T)',Sig_i');
    end
    [S, tmpP1, tmpP2] = sample_regimes(S, like, P, tmpP1, tmpP2, T);

        % sample P [121-129]
    P = sample_P(S, P, alp0, r);

    if isim>burnin                                    % [131-139]
        isave = isim-burnin;
        store_theta(isave,:) = theta(:);
        store_Sig(isave,:) = Sig(:);
        for j=1:r
            store_S(:,j) = store_S(:,j) + (S == j);
        end
        store_P(isave,:,:) = P;
    end

    if (mod(isim, pr.progress_every) == 0)            % [141-143]
        disp([num2str(isim) ' loops... '])
    end
end
disp(' ' );   % [147]

res = struct('k',k, 'm',m, 'bigX',bigX, 'prior',prior, ...
    'store_theta',store_theta, 'store_Sig',store_Sig, 'store_P',store_P, ...
    'store_S',store_S, 'count_empty',count_empty);
    % posterior summaries [148-150]
res.theta_hat = mean(store_theta)';
res.thetaCI = quantile(store_theta,[.05 .95])';
res.S_hat = store_S/nsims;
end

% -------------------------------------------------------------------------
function res = est_rs_r1(Y, shortY, X, T, n, p, r, nsims, burnin, pr)
% VAR_RS_R1.m functionized line-for-line: constant coefficients theta, error variances
% that switch with the regime. Draw order per sweep: theta -> Sig_i for each regime ->
% S -> the rows of P.
m = n*(n-1)/2;      % dimension of the impact matrix [8]
k = n^2*p + n + m;  % dimension of states [9]

    % prior [11-17]; the handle's third argument is the transition matrix
atheta = pr.prior_mean*ones(k,1); Vtheta = pr.prior_var*ones(k,1);
nu0 = pr.nu*ones(n,1); S0 = pr.S_sig*ones(n,1).*(nu0-1);
alp0 = pr.rs.alp0*ones(r,1);  % symmetric prior
cpri = -.5*k*log(2*pi) - .5*sum(log(Vtheta)) + r*nu0'*log(S0) - r*sum(gammaln(nu0));
prior = @(the,s,Pm) cpri -.5*(the-atheta)'*((the-atheta)./Vtheta) ...
    - (repmat(nu0,r,1)+1)'*log(s) - sum(repmat(S0,r,1)./s) + sum(ldiripdf(Pm,alp0)) ;

    % design [19-31]
X2 = zeros(n*T,n*(n-1)/2);
count = 0;
for j=2:n
    X2(j:n:end,count+1:count+j-1) = -shortY(:,1:j-1);
    count = count + j-1;
end
bigX = [bvar.util.surform2([ones(T,1) X],n) sparse(X2)];

    % initialize the Markov chain [33-50]
S = [kron((1:r-1)',ones(floor(T/r),1));r*ones(T-floor(T/r)*(r-1),1)];
P = triu(pr.rs.P_move*ones(r,r)/(r-1),1) + tril(pr.rs.P_move*ones(r,r)/(r-1),-1);
P = P + pr.rs.P_stay*eye(r);

theta = zeros(k,1);
Z = [ones(T,1) X];
tmptheta = (Z'*Z)\(Z'*shortY);
theta(1:n^2*p+n) = reshape(tmptheta',n^2*p+n,1);
Sig = zeros(n,r);
for i=1:r
    idx = (S == i);
    Ti = sum(idx);
    Z = [ones(Ti,1) X(idx,:)];
    shortYi = shortY(idx,:);
    E = shortYi - Z*tmptheta;
    Sig(:,i) = sum(E.^2)'/Ti;
end

    % initialize for storage [52-59]
store_Sig = zeros(nsims,n*r);
store_theta = zeros(nsims,k);
store_P = zeros(nsims,r,r);
store_S = zeros(T,r);
like = zeros(T,r);
tmpP1 = zeros(T,r);                   % p(s_t|Y_t,\theta,P)
tmpP2 = zeros(T,r); tmpP2(1,:) = pr.rs.p2_first; % p(s_t|Y_{t-1},\theta,P)

disp(['Starting VAR-RS-R1 with ' num2str(r) ' regimes.... ']);   % [63]
count_empty = 0;
for isim = 1:nsims + burnin
        % sample theta [67-72]
    iSig = 1./Sig;
    XiSig = bigX'*sparse(1:T*n,1:T*n,reshape(iSig(:,S),T*n,1));
    Ktheta = sparse(1:k,1:k,1./Vtheta) + XiSig*bigX;
    CKtheta = chol(Ktheta,'lower');
    theta_hat = CKtheta'\(CKtheta\(atheta./Vtheta + XiSig*Y));
    theta = theta_hat + CKtheta'\randn(k,1);

        % sample Sig [74-85]
    E = reshape(Y - bigX*theta,n,T)';
    for i=1:r
        idx = (S == i);
        Ti = sum(idx);
        if Ti == 0 % no active regimes, draw from prior
            count_empty = count_empty + 1;
            Sig(:,i) = 1./gamrnd(nu0,1./S0);
        else
            Sig(:,i) = 1./gamrnd(nu0+Ti/2, 1./(S0 + sum(E(idx,:).^2)'/2));
        end
    end

        % sample S [87-108]
    mu = bigX*theta;
    for i=1:r
        Sig_i = Sig(:,i);
        like(:,i) = mvnpdf(shortY,reshape(mu,n,T)',Sig_i');
    end
    [S, tmpP1, tmpP2] = sample_regimes(S, like, P, tmpP1, tmpP2, T);

        % sample P [110-118]
    P = sample_P(S, P, alp0, r);

    if isim>burnin                                    % [120-128]
        isave = isim-burnin;
        store_theta(isave,:) = theta;
        store_Sig(isave,:) = Sig(:);
        for j=1:r
            store_S(:,j) = store_S(:,j) + (S == j);
        end
        store_P(isave,:,:) = P;
    end

    if (mod(isim, pr.progress_every) == 0)            % [130-132]
        disp([num2str(isim) ' loops... '])
    end
end
disp(' ' );   % [137]

res = struct('k',k, 'm',m, 'bigX',bigX, 'prior',prior, ...
    'store_theta',store_theta, 'store_Sig',store_Sig, 'store_P',store_P, ...
    'store_S',store_S, 'count_empty',count_empty);
    % posterior summaries [139-141]
res.theta_hat = mean(store_theta)';
res.thetaCI = quantile(store_theta,[.05 .95])';
res.S_hat = store_S/nsims;
end

% -------------------------------------------------------------------------
function res = est_rs_r2(Y, shortY, X, T, n, p, r, nsims, burnin, pr)
% VAR_RS_R2.m functionized line-for-line: coefficients that switch with the regime,
% common error variances. Draw order per sweep: theta_i for each regime -> Sig -> S ->
% the rows of P. A regime with no observations redraws Sig from its prior inside the
% regime loop (quirk, see the header of this file).
m = n*(n-1)/2;      % dimension of the impact matrix [8]
k = n^2*p + n + m;  % dimension of states [9]

    % prior [11-18]; the handle's third argument is the transition matrix
atheta = pr.prior_mean*ones(k,1); Vtheta = pr.prior_var*ones(k,1);
nu0 = pr.nu*ones(n,1); S0 = pr.S_sig*ones(n,1).*(nu0-1);
alp0 = pr.rs.alp0*ones(r,1);  % symmetric prior

cpri = -.5*k*r*log(2*pi) - .5*r*sum(log(Vtheta)) + nu0'*log(S0) - sum(gammaln(nu0));
prior = @(the,s,Pm) cpri -.5*(the-repmat(atheta,r,1))'*((the-repmat(atheta,r,1))./repmat(Vtheta,r,1)) ...
    -(nu0+1)'*log(s) - sum(S0./s) + sum(ldiripdf(Pm,alp0)) ;

    % design [20-32]
X2 = zeros(n*T,n*(n-1)/2);
count = 0;
for j=2:n
    X2(j:n:end,count+1:count+j-1) = -shortY(:,1:j-1);
    count = count + j-1;
end
bigX = [bvar.util.surform2([ones(T,1) X],n) sparse(X2)];

    % initialize the Markov chain [34-51]
S = [kron((1:r-1)',ones(floor(T/r),1));r*ones(T-floor(T/r)*(r-1),1)];
P = triu(pr.rs.P_move*ones(r,r)/(r-1),1) + tril(pr.rs.P_move*ones(r,r)/(r-1),-1);
P = P + pr.rs.P_stay*eye(r);

theta = zeros(k,r);
sumE2 = zeros(n,1);
for i=1:r
    idx = (S == i);
    Ti = sum(idx);
    Z = [ones(Ti,1) X(idx,:)];
    shortYi = shortY(idx,:);
    tmptheta = (Z'*Z)\(Z'*shortYi);
    E = shortYi - Z*tmptheta;
    theta(1:n^2*p+n,i) = reshape(tmptheta',n^2*p+n,1);
    sumE2 = sumE2 + sum(E.^2)';
end
Sig = sumE2/T;

    % initialize for storage [53-60]
store_Sig = zeros(nsims,n);
store_theta = zeros(nsims,k*r);
store_P = zeros(nsims,r,r);
store_S = zeros(T,r);
like = zeros(T,r);
tmpP1 = zeros(T,r);                   % p(s_t|Y_t,\theta,P)
tmpP2 = zeros(T,r); tmpP2(1,:) = pr.rs.p2_first; % p(s_t|Y_{t-1},\theta,P)

disp(['Starting VAR-RS-R2 with ' num2str(r) ' regimes.... ']);   % [65]
count_empty = 0;
for isim = 1:nsims + burnin
        % sample theta [69-96]
    sumE2 = zeros(n,1);
    for i=1:r
        idx = (S == i);
        Ti = sum(idx);
        if Ti == 0 % not an active regime, draw from prior
            count_empty = count_empty + 1;
            theta(:,i) = atheta + sqrt(Vtheta).*randn(k,1);
            Sig = 1./gamrnd(nu0,1./S0);
        else
            shortYi = shortY(idx,:);   %#ok<NASGU> % dead in the legacy too [79]
            Yi = reshape(shortY(idx,:)',Ti*n,1);

            X2i = zeros(n*Ti,n*(n-1)/2);
            count = 0;
            for j=2:n
                X2i(j:n:end,count+1:count+j-1) = -shortY(idx,1:j-1);
                count = count + j-1;
            end
            bigXi = [bvar.util.surform2([ones(Ti,1) X(idx,:)],n) sparse(X2i)];
            XiSig = bigXi'*sparse(1:Ti*n,1:Ti*n,repmat(1./Sig,Ti,1));
            Ktheta = sparse(1:k,1:k,1./Vtheta) + XiSig*bigXi;
            CKtheta = chol(Ktheta,'lower');
            theta_hat = CKtheta'\(CKtheta\(atheta./Vtheta + XiSig*Yi));
            theta(:,i) = theta_hat + CKtheta'\randn(k,1);

            sumE2 = sumE2 + sum((reshape(Yi - bigXi*theta(:,i),n,Ti)').^2)';
        end
    end

        % sample Sig [98-99]
    Sig = 1./gamrnd(nu0+T/2, 1./(S0 + sumE2/2));

        % sample S [101-121]
    for i=1:r
        mu_i = bigX*theta(:,i);
        like(:,i) = mvnpdf(shortY,reshape(mu_i,n,T)',Sig');
    end
    [S, tmpP1, tmpP2] = sample_regimes(S, like, P, tmpP1, tmpP2, T);

        % sample P [123-131]
    P = sample_P(S, P, alp0, r);

    if isim>burnin                                    % [133-141]
        isave = isim-burnin;
        store_theta(isave,:) = theta(:);
        store_Sig(isave,:) = Sig(:);
        for j=1:r
            store_S(:,j) = store_S(:,j) + (S == j);
        end
        store_P(isave,:,:) = P;
    end

    if (mod(isim, pr.progress_every) == 0)            % [143-145]
        disp([num2str(isim) ' loops... '])
    end
end
disp(' ' );   % [150]

res = struct('k',k, 'm',m, 'bigX',bigX, 'prior',prior, ...
    'store_theta',store_theta, 'store_Sig',store_Sig, 'store_P',store_P, ...
    'store_S',store_S, 'count_empty',count_empty);
    % posterior summaries [152-154]
res.theta_hat = mean(store_theta)';
res.thetaCI = quantile(store_theta,[.05 .95])';
res.S_hat = store_S/nsims;
end

% -------------------------------------------------------------------------
function [S, tmpP1, tmpP2] = sample_regimes(S, like, P, tmpP1, tmpP2, T)
% the forward filter and backward draw of the regimes, identical in the three RS
% scripts (VAR_RS.m 104-119, VAR_RS_R1.m 93-108, VAR_RS_R2.m 106-121)
for t=1:T
    if t>1
        tmpP2(t,:) = tmpP1(t-1,:)*P;
    end
    tmpP1(t,:) = tmpP2(t,:) .* like(t,:);
    tmpP1(t,:) = tmpP1(t,:)/sum(tmpP1(t,:));
end
for t = T:-1:1
    if t == T
        S(t) = find(cumsum(tmpP1(t,:)) > rand,1);
    else
        prob = tmpP1(t,:)'.*P(:,S(t+1));
        prob = prob/sum(prob);
        S(t) = find(cumsum(prob) > rand,1);
    end
end
end

function P = sample_P(S, P, alp0, r)
% the Dirichlet draw of each row of P, identical in the three RS scripts
% (VAR_RS.m 122-129, VAR_RS_R1.m 111-118, VAR_RS_R2.m 124-131)
for i = 1:r
    ni = zeros(r,1);
    idx = find(S(1:end-1) == i);
    for j = 1:r
        ni(j) = sum(S(idx+1) == j);
    end
    P(i,:) = dirirnd(alp0+ni);
end
end

% -------------------------------------------------------------------------
function Xout = construct_x(X,X2,n)
% constructX.m, verbatim: the sparse regressor matrix of a TVP-VAR whose states stack
% the intercepts, the lag coefficients and the impact elements of each period; with X
% empty, the impact-element columns alone
k = n*(n-1)/2;
[T,m] = size(X);
if T == 0
    T = size(X2,1);
    X1 = [];
    idi1 = [];
    idj1 = [];
    idj2 = (1:T*k)';
else
    X1 = [ones(n*T,1) kron(X,ones(n,1))];
    m = m+1;
    tempid = reshape(1:T*(m*n+k),m*n+k,T)';
    idi1 = kron((1:n*T)',ones(m,1));
    idj1 = reshape(tempid(:,1:n*m)',T*m*n,1);
    idj2 = reshape(tempid(:,n*m+1:end)',T*k,1);
end
idi2 = zeros(T*k,1);
count = 0;
for i=2:n
    for j=1:i-1
        idi2(count+1:k:end) = i:n:T*n;
        count = count + 1;
    end
end
Xout = sparse([idi1; idi2],[idj1; idj2],[reshape(X1',n*T*m,1);reshape(X2',T*k,1)]);
end

function draws = dirirnd(alp,N)
% dirirnd.m, verbatim: N draws from the Dirichlet distribution with parameter alp
if nargin == 1
    N = 1;
end
n = length(alp);
x = gamrnd(repmat(alp',N,1),1,N,n);
draws = x./repmat(sum(x,2),1,n);
end

function lden = ldiripdf(y, alpha)
% ldiripdf.m, verbatim: the log Dirichlet density of each row of y
[~,k] = size(y);
if ~(k == length(alpha))
    error('dimensions do not match ');
end
if size(alpha, 1) < size(alpha, 2)
    alpha = alpha';
end

const = gammaln(sum(alpha)) - sum(gammaln(alpha));
lden = const + log(y)*(alpha - 1);
end
