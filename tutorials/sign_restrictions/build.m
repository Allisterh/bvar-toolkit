%% build - regenerate the figures and numbers of tutorials/sign_restrictions/README.md
%
% The 35-variable application of Chan, Matthes and Yu (2026): eight structural
% shocks identified by sign restrictions on the impact responses and by two
% ranking restrictions, on US quarterly data from 1983Q1 to 2019Q4. The model is
% a homoskedastic VAR with five lags under the asymmetric conjugate prior of
% Chan (2022), whose two shrinkage hyperparameters are chosen by maximizing the
% marginal likelihood, evaluated with the ridge of 1e-6 that this package adds
% to the posterior precision at n = 35.
%
% Identification is the exercise. A candidate rotation is admissible when every
% shock has at least one column of the rotated Cholesky factor that satisfies
% its restrictions, and bvar.structural.sign_assign then draws one assignment
% from those available. The accept-reject rule of Rubio-Ramirez, Waggoner and
% Zha (2010), bvar.structural.sign_restrict, instead requires column i to
% satisfy shock i. Both are applied to every candidate here, so the run reports
% what each rule accepts out of the same draws.
%
% The restrictions are written out below in a readable form and then checked
% against the matrices in the package's own driver, which this script reads.
%
% This build keeps 100 admissible draws where the paper keeps 1,000.
%
% The posterior summaries of the impulse responses go to irf_bands.mat and the
% figures are written next to this file. Everything printed goes to a log in
% tempdir.
%
% Usage, from anywhere:  run tutorials/sign_restrictions/build.m

tdir = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(tdir));
logf = fullfile(tempdir, 'bvar_sign_restrictions_build_log.txt');
if exist(logf, 'file'), delete(logf); end
diary(logf);
fprintf('tutorials/sign_restrictions/build.m, %s, MATLAB %s\n', ...
    char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm')), version);
t_all = tic;
run(fullfile(repo, 'setup.m'));

%% ---- settings ----
p = 5;  n0 = 8;  nkeep = 100;  nbatch = 1000;  horizon = 36;
seed = 20260920;  ridge = 1e-6;  k0 = [.04 .0016];
pkg = fullfile(repo, 'replications', 'chan_matthes_yu2026_qe_svarsign', 'legacy');

%% ---- data ----
% The driver reads this block of the csv with xlsread; readmatrix returns the
% same 148 x 35 array up to the last bit of one cell, so a run here is not
% bitwise comparable with a legacy run.
csv = fullfile(pkg, 'data', 'data35_1975.csv');
data = readmatrix(csv, 'Range', 'B35:AJ182');
lines = readlines(csv);
vars = strtrim(split(lines(1), ','));
vars = vars(2:36)';
dates = datetime(1983,1,1) + calquarters(0:size(data,1)-1);
[nobs, n] = size(data);
assert(nobs == 148 && n == 35, 'the data block is not 148 x 35');
Y0 = data(1:n0,:);  Y = data(n0+1:end,:);
[T, ~] = size(Y);
[~, Z] = bvar.util.build_lags([Y0(end-p+1:end,:); Y], p);
idx_ns = 1:n;                       % every series enters in levels

fprintf('\n%d variables, %d quarters (%s to %s), %d lags\n', n, nobs, ...
    string(dates(1),'uuuuQQQ'), string(dates(end),'uuuuQQQ'), p);
fprintf('%d observations after %d initial conditions\n', T, n0);

%% ---- the prior, with its shrinkage chosen by marginal likelihood ----
t0 = tic;
[ml_opt, kappa] = bvar.priors.acp_opt_kappa(Y0, Y, Z, p, k0, 'redu', idx_ns, 'ridge', ridge);
sig2 = bvar.priors.resid_var_ar4(Y0, Y);
prior = bvar.priors.acp_redu(n, p, kappa, sig2, idx_ns);
fprintf('\nshrinkage by marginal likelihood: kappa1 %.4g (own lags), kappa2 %.4g (cross lags)\n', ...
    kappa(1), kappa(2));
fprintf('log marginal likelihood %.1f, chosen in %.1f minutes\n', ml_opt, toc(t0)/60);

%% ---- the restrictions ----
shock = ["demand" "investment" "financial" "monetary" "government spending" ...
         "technology" "labor supply" "wage bargaining"];
m = numel(shock);
pos = {[1 10 11 12 13 14 20 21 25 26 30], ...
       [1 10 11 12 13 14 20 21 25 26 30], ...
       [1 10 11 12 13 14 20 21 25 26 30 34], ...
       [19 25 26 27 28 29 30 31 32], ...
       [1 7 9 10 11 12 13 14 25 26 30], ...
       [1 2 4 15 16 17], ...
       [1 19], ...
       1};
neg = {19, [19 34], 19, [1 10 11 12 13 14 18 20 21 34], [8 19], ...
       [10 11 12 13 14 19], [10 11 12 13 14 15], [10 11 12 13 14 15 19]};
S = nan(n, m);
for i = 1:m, S(pos{i},i) = 1;  S(neg{i},i) = -1; end

% Each row is a linear combination of impact responses required to be <= 0 for
% that shock's column. Variables 1, 4 and 7 are output, nonresidential
% investment and government spending.
kr = 2;
Rineq = zeros(m, n, kr);
Rineq(1,[1 4],1) = [-1 1];      % demand:     investment rises by less than output
Rineq(2,[1 4],1) = [1 -1];      % investment: investment rises by more
Rineq(3,[1 4],1) = [1 -1];      % financial:  investment rises by more
Rineq(1,[1 7],2) = [-1 1];      % demand:     government spending rises by less
Rineq(2,[1 7],2) = [-1 1];      % investment: government spending rises by less
Rineq(3,[1 7],2) = [-1 1];      % financial:  government spending rises by less
Rineq(5,[1 7],2) = [1 -1];      % government: government spending rises by more

[S_pkg, R_pkg] = legacy_restrictions(pkg);
assert(isequaln(S, S_pkg), 'the sign restrictions differ from the package''s driver');
assert(isequal(Rineq, R_pkg), 'the ranking restrictions differ from the package''s driver');
fprintf('\n%d sign restrictions and %d ranking restrictions over %d shocks,\n', ...
    sum(~isnan(S(:))), sum(any(Rineq ~= 0, 2), 'all'), m);
fprintf('checked against %s\n', 'main_35VAR_Figs2to10_except5.m');

[sep, unsep] = bvar.structural.check_separable(S, Rineq);
assert(sep, 'the restrictions leave %d pairs of shocks unseparated', size(unsep,1));
[~, by_rank] = bvar.structural.check_separable(S);
fprintf('all %d pairs of shocks are separated, %d of them only by the ranking restrictions\n', ...
    m*(m-1)/2, size(by_rank,1));

% the same ranking restrictions in the form sign_restrict takes: one row per
% restriction, with the column of L it applies to
Rrows = zeros(0, n);  Ridx = zeros(0, 1);
for i = 1:m
    for j = 1:kr
        if any(Rineq(i,:,j) ~= 0)
            Rrows(end+1,:) = Rineq(i,:,j);   %#ok<SAGROW>
            Ridx(end+1,1) = i;               %#ok<SAGROW>
        end
    end
end

%% ---- draw until 100 rotations are admissible ----
fprintf('\ndrawing until %d rotations are admissible, %d posterior draws per batch\n', ...
    nkeep, nbatch);
rng(seed, 'twister');
resp = zeros(nkeep, n, m, horizon);
nsat = 0;  ntotal = 0;  nstrict = 0;  ibatch = 0;
t0 = tic;
while nsat < nkeep
    ibatch = ibatch + 1;
    [alp, beta, sig] = bvar.samplers.acp_theta_sig(Y0, Y, p, prior, nbatch);
    [Btilde, Sigtilde] = bvar.structural.reduced_form(alp, beta, sig);
    for isim = 1:nbatch
        ntotal = ntotal + 1;
        L0 = chol(squeeze(Sigtilde(isim,:,:)), 'lower');
        Q = bvar.structural.qr_sign(randn(n));
        L = L0*Q;
        nstrict = nstrict + bvar.structural.sign_restrict(L, S, Rrows, Ridx);
        [ok, Lo] = bvar.structural.sign_assign(L, S, Rineq, kr);
        if ok && nsat < nkeep
            nsat = nsat + 1;
            B = reshape(Btilde(isim,:), n*p+1, n);
            resp(nsat,:,:,:) = bvar.structural.irf_redu(B(2:end,:), Lo, horizon, m);
        end
    end
    if mod(ibatch, 25) == 0
        fprintf('  %d admissible out of %d candidates, %.1f minutes\n', ...
            nsat, ntotal, toc(t0)/60);
    end
end
t_draw = toc(t0);

fprintf('\n%d admissible draws from %d candidate rotations, %.0f candidates each,\n', ...
    nsat, ntotal, ntotal/nsat);
fprintf('in %.1f minutes. The paper reports about 5,500 candidates per draw.\n', t_draw/60);
fprintf('the accept-reject rule of Rubio-Ramirez, Waggoner and Zha accepted %d of the same %d.\n', ...
    nstrict, ntotal);
fprintf('it tests one assignment of shocks to columns out of the %.1e available;\n', ...
    prod(n-m+1:n));
fprintf('both rules are free to flip the sign of a column, so the sign is not the difference.\n');

%% ---- the impulse responses ----
lo = squeeze(quantile(resp, .16, 1));
md = squeeze(median(resp, 1));
hi = squeeze(quantile(resp, .84, 1));
save(fullfile(tdir, 'irf_bands.mat'), 'lo', 'md', 'hi', 'vars', 'shock', ...
    'kappa', 'ml_opt', 'nsat', 'ntotal', 'nstrict', 'horizon', 'p', 'seed');

show = [1 11 25 4 19 15];
label = ["GDP" "PCE price index" "Federal funds rate" "Nonresidential investment" ...
         "Unemployment rate" "Real compensation"];
short = ["GDP" "prices" "fed funds" "investment" "unemployment" "compensation"];
fprintf('\nimpact response to a one-standard-deviation shock, posterior median\n');
fprintf('  %-21s', 'shock');  fprintf('%13s', short);  fprintf('\n');
for is = 1:m
    fprintf('  %-21.21s', shock(is));
    fprintf('%13.3f', md(show,is,1));
    fprintf('\n');
end

for is = [4 3]          % the monetary and financial shocks get a figure each
    figure('Position', [100 100 780 460]);
    for j = 1:numel(show)
        subplot(2, 3, j);
        hold on
        bvar.util.shaded_band((0:horizon-1)', squeeze(lo(show(j),is,:)), ...
            squeeze(hi(show(j),is,:)), .85);
        plot(0:horizon-1, squeeze(md(show(j),is,:)), 'k--', 'LineWidth', 1.1);
        yline(0, 'k:');
        hold off; box off; xlim([0 horizon-1]);
        title(label(j));
    end
    sgtitle(sprintf('%s shock', shock(is)));
    exportgraphics(gcf, fullfile(tdir, sprintf('fig_irf_%s.png', ...
        regexprep(shock(is), '\s+', '_'))), 'Resolution', 150);
end

fprintf('\nbuild finished in %.1f minutes\n', toc(t_all)/60);
diary off

%% -------------------------------------------------------------------------
function [S, Rineq] = legacy_restrictions(pkg)
% The package states its restrictions inline in a driver script, so they are read
% from that script and evaluated here. The block sets S, Rineq, m and k; the
% trailing 'end' it does not include closes an if.
src = fileread(fullfile(pkg, 'main_35VAR_Figs2to10_except5.m'));
i0 = strfind(src, 'demand = [');
i1 = strfind(src, 'start_time = clock');
assert(isscalar(i0) && isscalar(i1), 'the restriction block was not found in the driver');
n = 35;            %#ok<NASGU>
S = [];  Rineq = [];        % both are assigned by the eval below
eval(regexprep(src(i0:i1-1), 'end\s*$', ''));
end
