%% your_data - choose the shrinkage hyperparameters of a BVAR on your data
%
% Set the file, the columns, their names, which of them are nonstationary and the
% lag lengths below, then run the script. It chooses the own-lag and other-lag
% shrinkage kappa2 and kappa3 of the asymmetric conjugate prior by maximizing the
% closed-form marginal likelihood, compares the result with the best symmetric
% prior (kappa2 = kappa3) and with the subjective values kappa2 = 0.04 and
% kappa3 = 0.0016, repeats the choice for each lag length up to pmax, and plots
% the marginal likelihood around the optimum. The prior is set on the
% structural-form coefficients, as in Section 12.3 of Bayesian Macroeconometrics.
% The defaults use the 21-variable dataset of the tutorial and take a few seconds.
%
% Give cols as column numbers, which are read with readmatrix and count every
% column of the file, or as column names, which are read with readtable. The
% selected columns must have no missing values inside the sample; rows missing at
% either end are dropped. Setting datecol checks that the rows kept are evenly
% spaced. The series must be stationary, transformed as needed; that choice is
% separate from nonstationary below, which only centers the prior.
%
% The first n0 rows serve as initial conditions: at least max(pmax,4) of them.

repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
run(fullfile(repo, 'setup.m'))

%% ---- settings ----
file    = fullfile(repo, 'replications', 'chan2019wp_acp', 'legacy', 'macrodata_Q_2018Q4.csv');
cols    = [1 2 18 22 34 35 57 59 76 81 95 97 120 123 133 138 145 148 152 160 245];
names   = ["GDP" "Consumption" "Disposable income" "Industrial production" ...
           "Capacity utilization" "Payroll employment" "Civilian employment" ...
           "Unemployment rate" "Hours" "Housing starts" "PCE prices" "GDP deflator" "CPI" ...
           "PPI" "Real earnings" "Productivity" "3-month T-bill" "10-year yield" ...
           "Baa spread" "Real M1" "S&P 500"];
datecol = [];       % the column holding the dates, by number or name; [] to skip the check
nonstationary = []; % positions in cols whose first own lag gets prior mean one
p       = 4;        % lags
pmax    = 8;        % longest lag length in the scan
n0      = 8;        % rows used as initial conditions

%% ---- data ----
if isnumeric(cols)
    raw = readmatrix(file);                   % column numbers count every column of the file
    sel = raw(:, cols);
    dates = [];
    if ~isempty(datecol), dates = raw(:, datecol); end
else
    tbl = readtable(file, 'VariableNamingRule', 'preserve');
    sel = tbl{:, cols};
    dates = [];
    if ~isempty(datecol), dates = tbl{:, datecol}; end
end
assert(numel(names) == numel(cols), 'names and cols must have the same length');
assert(n0 >= max(pmax, 4), 'n0 must be at least max(pmax,4)');
keep = all(isfinite(sel), 2);
assert(any(keep), 'every row of the selected columns has a missing value');
lo = find(keep, 1);  hi = find(keep, 1, 'last');
assert(all(keep(lo:hi)), 'the selected columns have missing values inside the sample');
data = sel(lo:hi, :);
flat = std(data) == 0;
assert(~any(flat), 'these series are constant: %s', strjoin(names(flat), ', '));
if ~isempty(dates)
    d = dates(lo:hi);
    if isdatetime(d)
        step = days(diff(d));
        even = all(step > 0) && max(abs(step - median(step))) <= 3;   % months vary by up to 3 days
    else
        step = diff(double(d));
        even = all(step > 0) && max(abs(step - median(step))) <= 1e-6*max(1, abs(median(step)));
    end
    assert(even, 'the dates of the rows kept are not evenly spaced');
end
Y0 = data(1:n0, :);
Y  = data(n0+1:end, :);
[T, n] = size(Y);
fprintf('\n%d variables, %d observations after %d initial conditions\n', n, T, n0);

%% ---- the three priors at p lags ----
[~, Z] = bvar.util.build_lags([Y0(end-p+1:end,:); Y], p);
sig2 = bvar.priors.resid_var_ar4(Y0, Y);
[ml_a, ka] = bvar.priors.acp_opt_kappa(Y0, Y, Z, p, [.04 .04], 'stru', nonstationary);
[ml_s, ks] = bvar.priors.acp_opt_kappa(Y0, Y, Z, p, [], 'stru', nonstationary, 'symmetric', true);
ml_0 = bvar.ml.acp(p, Y, Z, bvar.priors.acp_stru(n, p, [.04 .0016 1 100], sig2, nonstationary));
    % the symmetric optimum is feasible under the asymmetric prior, so a search
    % started there should reach the same maximum
[ml_b, kb] = bvar.priors.acp_opt_kappa(Y0, Y, Z, p, [ks(1) ks(1)], 'stru', nonstationary);
gap = ml_b - ml_a;
if gap > 0, ml_a = ml_b; ka = kb; end
fprintf('\nlog marginal likelihood at p = %d\n', p);
fprintf('%-42s %10s %10s %10s %10s\n', '', 'kappa2', 'kappa3', 'log ML', 'difference');
fprintf('%-42s %10.4f %10.4f %10.1f %10.1f\n', 'symmetric, chosen by marginal likelihood', ks(1), ks(2), ml_s, ml_s - ml_a);
fprintf('%-42s %10.4f %10.4f %10.1f %10.1f\n', 'subjective', .04, .0016, ml_0, ml_0 - ml_a);
fprintf('%-42s %10.4f %10.4f %10.1f %10.1f\n', 'asymmetric, chosen by marginal likelihood', ka(1), ka(2), ml_a, 0);
fprintf(['kappa2 and kappa3 are the own-lag and other-lag shrinkage of Section 12.3 of\n' ...
    'Bayesian Macroeconometrics; bvar.priors.acp_opt_kappa returns them as kappa(1:2).\n']);
fprintf('two starting values for the asymmetric search, (0.04, 0.04) and the symmetric\n');
fprintf('optimum (%.4f, %.4f), reach log marginal likelihoods %.4f apart\n', ks(1), ks(2), abs(gap));
fprintf('the estimates condition on these hyperparameters: their uncertainty is left out\n');

%% ---- the lag length ----
fprintf('\nlag length, with kappa2 and kappa3 chosen again for each\n');
fprintf('%4s %10s %10s %12s %14s\n', 'p', 'kappa2', 'kappa3', 'log ML', 'symmetric ML');
lml = zeros(pmax, 1);
for pp = 1:pmax
    [~, Zp] = bvar.util.build_lags([Y0(end-pp+1:end,:); Y], pp);
    [lml(pp), kp] = bvar.priors.acp_opt_kappa(Y0, Y, Zp, pp, [.04 .04], 'stru', nonstationary);
    lsym = bvar.priors.acp_opt_kappa(Y0, Y, Zp, pp, [], 'stru', nonstationary, 'symmetric', true);
    fprintf('%4d %10.4f %10.5f %12.1f %14.1f\n', pp, kp(1), kp(2), lml(pp), lsym);
end
[~, pbest] = max(lml);
fprintf('highest marginal likelihood at p = %d\n', pbest);
if pbest == pmax
    fprintf(['this is the longest lag length in the scan: raise pmax, and n0 with it,\n' ...
        'to see whether the marginal likelihood keeps rising\n']);
end

%% ---- the marginal likelihood around the optimum ----
    % a logarithmic grid from a tenth to ten times the optimum, widened to show
    % the symmetric and subjective priors
r2 = [min([ka(1)/10, ks(1), .04])/1.5, ka(1)*10];
r3 = [min([ka(2)/10, .0016])/1.5, max([ka(2)*10, ks(2)])*1.5];
g2 = logspace(log10(r2(1)), log10(r2(2)), 41);
g3 = logspace(log10(r3(1)), log10(r3(2)), 41);
[K2, K3] = meshgrid(g2, g3);
L = zeros(size(K2));
for i = 1:numel(K2)
    L(i) = bvar.ml.acp(p, Y, Z, bvar.priors.acp_stru(n, p, [K2(i), K3(i), 1, 100], sig2, nonstationary));
end
figure; hold on; box off
contour(K2, K3, exp(L - max(L(:))), 0.1:0.1:0.9, 'LineWidth', 1);
set(gca, 'XScale', 'log', 'YScale', 'log');
lo = max(g2(1), g3(1));  hi = min(g2(end), g3(end));
if lo < hi, plot([lo hi], [lo hi], 'k--'); end
h = [plot(ka(1), ka(2), 'kp', 'MarkerFaceColor', 'w', 'MarkerSize', 11), ...
     plot(ks(1), ks(2), 'ko', 'MarkerFaceColor', 'k'), ...
     plot(.04, .0016, 'ks', 'MarkerFaceColor', 'k')];
xlim(r2); ylim(r3);
legend(h, {'asymmetric prior', 'symmetric prior', 'subjective prior'}, 'Location', 'best', 'Box', 'off');
xlabel('\kappa_2 (own lags), smaller is tighter'); ylabel('\kappa_3 (other lags), smaller is tighter');
title(sprintf('Marginal likelihood relative to its maximum, p = %d', p));
