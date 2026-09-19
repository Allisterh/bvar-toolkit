%% your_data - choose the shrinkage hyperparameters of a BVAR on your data
%
% Set the file, the columns, their names, which of them enter in levels and the
% lag lengths below, then run the script. It chooses the own-lag and other-lag
% shrinkage kappa2 and kappa3 of the asymmetric conjugate prior by maximizing the
% closed-form marginal likelihood, compares the result with the best symmetric
% prior (kappa2 = kappa3) and with the subjective values kappa2 = 0.04 and
% kappa3 = 0.0016, repeats the choice for each lag length up to pmax, and plots
% the marginal likelihood around the optimum. The prior is set on the
% structural-form coefficients, as in Section 12.3 of Bayesian Macroeconometrics.
% The defaults use the 21-variable dataset of the tutorial and take a few seconds.
%
% The file is read with readmatrix, and rows and columns without numbers, such as
% header rows or a date column, are dropped. The selected columns must have no
% missing values. Variables in levels get a prior mean of one on their first own
% lag, the others a prior mean of zero. The first n0 rows serve as initial
% conditions: at least max(pmax,4) of them.

repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
run(fullfile(repo, 'setup.m'))

%% ---- settings ----
file   = fullfile(repo, 'replications', 'chan2019wp_acp', 'legacy', 'macrodata_Q_2018Q4.csv');
cols   = [1 2 18 22 34 35 57 59 76 81 95 97 120 123 133 138 145 148 152 160 245];
names  = ["GDP" "Consumption" "Disposable income" "Industrial production" ...
          "Capacity utilization" "Payroll employment" "Civilian employment" ...
          "Unemployment rate" "Hours" "Housing starts" "PCE prices" "GDP deflator" "CPI" ...
          "PPI" "Real earnings" "Productivity" "3-month T-bill" "10-year yield" ...
          "Baa spread" "Real M1" "S&P 500"];
levels = [];                                   % positions in cols of the variables in levels
p      = 4;                                    % lags
pmax   = 8;                                    % longest lag length in the scan
n0     = 8;                                    % rows used as initial conditions

%% ---- data ----
raw = readmatrix(file);
raw = raw(any(~isnan(raw), 2), :);
raw = raw(:, any(~isnan(raw), 1));
data = raw(:, cols);
assert(~any(isnan(data(:))), 'the selected columns have missing values');
assert(n0 >= max(pmax, 4), 'n0 must be at least max(pmax,4)');
Y0 = data(1:n0, :);
Y  = data(n0+1:end, :);
[T, n] = size(Y);
fprintf('\n%d variables, %d observations after %d initial conditions\n', n, T, n0);

%% ---- the three priors at p lags ----
[~, Z] = bvar.util.build_lags([Y0(end-p+1:end,:); Y], p);
sig2 = bvar.priors.resid_var_ar4(Y0, Y);
[ml_a, ka] = bvar.priors.acp_opt_kappa(Y0, Y, Z, p, [.04 .04], 'stru', levels);
[ml_s, ks] = bvar.priors.acp_opt_kappa(Y0, Y, Z, p, [], 'stru', levels, 'symmetric', true);
ml_0 = bvar.ml.acp(p, Y, Z, bvar.priors.acp_stru(n, p, [.04 .0016 1 100], sig2, levels));
fprintf('\nlog marginal likelihood at p = %d\n', p);
fprintf('%-42s %10s %10s %10s %10s\n', '', 'kappa2', 'kappa3', 'log ML', 'difference');
fprintf('%-42s %10.4f %10.4f %10.1f %10.1f\n', 'symmetric, chosen by marginal likelihood', ks(1), ks(2), ml_s, ml_s - ml_a);
fprintf('%-42s %10.4f %10.4f %10.1f %10.1f\n', 'subjective', .04, .0016, ml_0, ml_0 - ml_a);
fprintf('%-42s %10.4f %10.4f %10.1f %10.1f\n', 'asymmetric, chosen by marginal likelihood', ka(1), ka(2), ml_a, 0);
fprintf(['kappa2 and kappa3 are the own-lag and other-lag shrinkage of Section 12.3 of\n' ...
    'Bayesian Macroeconometrics; bvar.priors.acp_opt_kappa returns them as kappa(1:2).\n']);

%% ---- the lag length ----
fprintf('\nlag length, with kappa2 and kappa3 chosen again for each\n');
fprintf('%4s %10s %10s %12s %14s\n', 'p', 'kappa2', 'kappa3', 'log ML', 'symmetric ML');
lml = zeros(pmax, 1);
for pp = 1:pmax
    [~, Zp] = bvar.util.build_lags([Y0(end-pp+1:end,:); Y], pp);
    [lml(pp), kp] = bvar.priors.acp_opt_kappa(Y0, Y, Zp, pp, [.04 .04], 'stru', levels);
    lsym = bvar.priors.acp_opt_kappa(Y0, Y, Zp, pp, [], 'stru', levels, 'symmetric', true);
    fprintf('%4d %10.4f %10.5f %12.1f %14.1f\n', pp, kp(1), kp(2), lml(pp), lsym);
end
[~, pbest] = max(lml);
fprintf('highest marginal likelihood at p = %d\n', pbest);

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
    L(i) = bvar.ml.acp(p, Y, Z, bvar.priors.acp_stru(n, p, [K2(i), K3(i), 1, 100], sig2, levels));
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
xlabel('\kappa_2 (own lags)'); ylabel('\kappa_3 (other lags)');
title(sprintf('Marginal likelihood relative to its maximum, p = %d', p));
