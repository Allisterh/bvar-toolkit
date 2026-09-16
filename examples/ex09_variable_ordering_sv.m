%% ex09 - Variable ordering in a VAR with stochastic volatility
%
% BOOK: Chapter 13, Section 13.1.3, and Chapter 14, Section 14.3, in Bayesian
% Macroeconometrics: Methods and Applications (Chapman & Hall/CRC, forthcoming).
%
% Two models of the reduced-form covariance, each estimated in the published
% variable order and in the reverse order:
%
%   Cholesky SV, as in ex04:  Sigma_t = A^{-1} diag(exp(h_t)) A^{-T},
%                             A unit lower triangular
%   Order-invariant SV:       Sigma_t = B0^{-1} diag(exp(h_t)) B0^{-T},
%                             B0 unrestricted, log-volatilities zero-mean
%
% Each model is also run a second time in the published order under another
% seed, which measures the Monte Carlo error the ordering gap is read against.
%
% DATA. replications/chan_koop_yu2024_jbes_oisv/legacy/FRED_MD_20vars.csv,
% read-only: monthly FRED-MD, 1959:03 to 2019:12, columns 4, 6, 12 and 13 -
% industrial production, the unemployment rate, PCE inflation and the federal
% funds rate, which the paper ranks first among its 20 variables. Thirteen
% lags, the first 24 months as initial conditions, and the prior constants of
% the package's preset.m. The chains are 1000 draws after 200 burn-in, against
% the published 30,000 after 5,000.
%
% See:
% Chan, J.C.C., Koop, G. and Yu, X. (2024). Large Order-Invariant Bayesian
% VARs with Stochastic Volatility, Journal of Business and Economic
% Statistics, 42(2): 825-837.
% Cogley, T. and Sargent, T.J. (2005). Drifts and Volatilities: Monetary
% Policies and Outcomes in the Post WWII US, Review of Economic Dynamics,
% 8(2): 262-302.
% McCracken, M.W. and Ng, S. (2016). FRED-MD: A Monthly Database for
% Macroeconomic Research, Journal of Business and Economic Statistics,
% 34(4): 574-589.

run(fullfile(fileparts(fileparts(mfilename('fullpath'))),'setup.m'))

repo = fileparts(fileparts(mfilename('fullpath')));
fprintf('\n=== ex09: variable ordering in a VAR with stochastic volatility ===\n');

%% ------------------------------------------------------------------
%  1. Data, the two orderings, and the settings
%  ------------------------------------------------------------------
pkg = fullfile(repo, 'replications', 'chan_koop_yu2024_jbes_oisv');
od = cd(pkg); guard = onCleanup(@() cd(od));     % preset.m resolves in its own folder
pr = preset();
clear guard

data = load(fullfile(pkg, 'legacy', pr.data_file));   % never written to; legacy is frozen
var_id = pr.var_id1;                                  % [4 6 12 13], the paper's first four
vnames = ["IP" "unemployment" "PCE inflation" "fed funds"];
p = pr.p;
n0 = pr.n0;

nsim = 1000; burnin = 200;
seeds = [20260915, 31415926];

n = numel(var_id);
fprintf('\nvariables, in the published order: %s\n', strjoin(cellstr(vnames), ', '));
fprintf('%d monthly observations from 1959:03, %d as initial conditions, p = %d lags\n', ...
    size(data,1), n0, p);
fprintf('%d draws after %d burn-in, two seeds\n', nsim, burnin);

%% ------------------------------------------------------------------
%  2. Six chains: two models x two orderings, plus a second seed for each
%     model in the published order
%  ------------------------------------------------------------------
cfg = struct('model', {'OI','OI','OI','CS','CS','CS'}, ...
             'rev',   {false, true, false, false, true, false}, ...
             'seed',  {seeds(1), seeds(1), seeds(2), seeds(1), seeds(1), seeds(2)}, ...
             'label', {'OI order 1','OI order 2','OI order 1, seed 2', ...
                       'CS order 1','CS order 2','CS order 1, seed 2'});

res = cell(1, numel(cfg));
for ic = 1:numel(cfg)
    ord = var_id;
    if cfg(ic).rev, ord = ord(end:-1:1); end
    Y0 = data(1:n0, ord);
    Y  = data(n0+1:end, ord);
    [~, X] = bvar.util.build_lags([Y0(end-p+1:end,:); Y], p);

    fprintf('\n%-20s ', cfg(ic).label);
    t0 = tic;
    rng(cfg(ic).seed, 'twister');
    res{ic} = mcmc(cfg(ic).model, Y, X, Y0, p, nsim, burnin, pr);
    if cfg(ic).rev                       % back to the published order for comparison
        res{ic}.Sig_mean = res{ic}.Sig_mean(:, end:-1:1, end:-1:1);
    end
    fprintf('%5.0f s\n', toc(t0));
end

%% ------------------------------------------------------------------
%  3. Every element of Sigma_t, against the two-seed yardstick
%  ------------------------------------------------------------------
%  Variances are compared as a share of their level. Correlations are compared
%  in correlation points, because a correlation near zero can change sign while
%  moving very little.
vpath = @(S,i) S(:,i,i);
cpath = @(S,i,j) S(:,i,j)./sqrt(S(:,i,i).*S(:,j,j));
relgap = @(a,b) 100*mean(abs(a-b)) / mean(abs(a)+abs(b))*2;
absgap = @(a,b) mean(abs(a-b));
col = [4 5; 4 6; 1 2; 1 3];                    % CS orderings, CS seeds, OI orderings, OI seeds

fprintf('\ndifference between two posterior-mean paths\n');
fprintf('%-34s %11s %11s %11s %11s\n', 'variances, as a share of their level', ...
    'CS orders', 'CS seeds', 'OI orders', 'OI seeds');
for ii = 1:n
    fprintf('  %-32s', sprintf('var(%s)', vnames(ii)));
    for ic = 1:4
        fprintf(' %10.1f%%', relgap(vpath(res{col(ic,1)}.Sig_mean, ii), vpath(res{col(ic,2)}.Sig_mean, ii)));
    end
    fprintf('\n');
end

prs = nchoosek(1:n, 2);
gap = zeros(size(prs,1), 4);
fprintf('%-34s\n', 'correlations, in correlation points');
for ip = 1:size(prs,1)
    i = prs(ip,1); j = prs(ip,2);
    for ic = 1:4
        gap(ip,ic) = absgap(cpath(res{col(ic,1)}.Sig_mean, i, j), cpath(res{col(ic,2)}.Sig_mean, i, j));
    end
    fprintf('  %-32s %11.3f %11.3f %11.3f %11.3f\n', ...
        sprintf('corr(%s, %s)', vnames(i), vnames(j)), gap(ip,:));
end

[~, ip] = max(gap(:,1));                        % where the ordering moves CS most
i = prs(ip,1); j = prs(ip,2);
fprintf(['\nreversing the ordering moves the CS correlation between %s and %s by %.3f on\n' ...
    'average, against %.3f between two seeds of the same CS run. The same correlation\n' ...
    'under OI moves %.3f with the ordering and %.3f between seeds.\n'], ...
    vnames(i), vnames(j), gap(ip,1), gap(ip,2), gap(ip,3), gap(ip,4));

%% ------------------------------------------------------------------
%  4. The two panels of Figures 8 and 9, on four variables
%  ------------------------------------------------------------------
T = size(res{1}.Sig_mean, 1);
dates = datetime(1961,3,1) + calmonths(0:T-1);
lbl = {cfg([4 5 1 2]).label};
figure('Name', 'ex09: two orderings, two models');
subplot(2,1,1); hold on; box off
for ic = [4 5 1 2], plot(dates, vpath(res{ic}.Sig_mean, 4), 'LineWidth', 1); end
title('variance of the federal funds rate'); legend(lbl, 'Location', 'northeast'); hold off
subplot(2,1,2); hold on; box off
for ic = [4 5 1 2], plot(dates, cpath(res{ic}.Sig_mean, i, j), 'LineWidth', 1); end
title(sprintf('correlation between %s and %s', vnames(i), vnames(j))); hold off
drawnow

fprintf('\nThe order-invariant model is drawn under two orderings that are the same model,\n');
fprintf('so its two paths differ only by Monte Carlo error. The Cholesky model is drawn\n');
fprintf('under two orderings that are two different models.\n');

%% ------------------------------------------------------------------
%  The sampler. Both models share the log-volatility and shrinkage blocks and
%  differ in the impact matrix and the coefficient draw, which is where the
%  ordering enters.
%  ------------------------------------------------------------------
function res = mcmc(model, Y, X, Y0, p, nsim, burnin, pr)
[T, n] = size(Y);
k = 1 + n*p;
is_oi = strcmp(model, 'OI');

    % Minnesota second moments, built from AR(4) residual variances
sig2_ar = bvar.priors.resid_var_ar4(Y0, Y);
[C, idx_kappa1, idx_kappa2] = bvar.priors.minnesota_C(n, p, sig2_ar);

    % priors: the preset values of the package, at this n
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
kappa = pr.oi.kappa_init;                    % [.1 .1 NaN 100]
if ~is_oi, kappa = pr.cs.kappa_init; end     % [.1 .1 1 100]

    % chain init
A = (X'*X + pr.ls_ridge*speye(k))\(X'*Y);
U = Y - X*A;
Sig_hat = U'*U/T;
h = repmat(log(diag(Sig_hat))', T, 1);
sig2 = 1./gamrnd(Hyper.nuh, 1./Hyper.Sh);
phi = min(Hyper.phi0 + sqrt(Hyper.Vphi).*randn(n,1), pr.phi_init_bnd);
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
store_kappa = zeros(nsim, 2);
for isim = 1:nsim + burnin
        % Vbeta scales with sig2, which holds the log-volatility state variances
        % at this point, as in the published samplers. The same rule applies in
        % every configuration, so it does not enter the comparison.
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
        ystar = log(E(:,ii).^2 + pr.sv_offset);
        if is_oi
            h(:,ii) = bvar.sv.ksc_ar1_mean(ystar, h(:,ii), pr.oi.h_mean_in_sv, phi(ii), sig2(ii));
        else
            h(:,ii) = bvar.sv.ksc_ar1_mean(ystar, h(:,ii), mu(ii), phi(ii), sig2(ii));
        end
    end
    if is_oi
        [phi, sig2] = bvar.sv.sv0_params(h, phi, Hyper);
    else
        [mu, phi, sig2] = bvar.sv.sv_params(h, mu, phi, Hyper);
    end

    [psi1, psi2, z_psi1, z_psi2, kappa, z_kappa] = bvar.samplers.horseshoe_kappa_psi( ...
        theta, idx_kappa1, idx_kappa2, C, kappa, z_psi1, z_psi2, z_kappa);
    Psi(idx_kappa1) = psi1;
    Psi(idx_kappa2) = psi2;

    if isim > burnin
        store_kappa(isim - burnin, :) = kappa(1:2);
        if is_oi
            Sig_sum = Sig_sum + bvar.structural.construct_Sigt(h, B0);
        else
            Sig_sum = Sig_sum + bvar.structural.construct_Sigt(h, Atri);
        end
    end
end

res.Sig_mean = Sig_sum/nsim;
res.kappa_mean = mean(store_kappa)';
end
