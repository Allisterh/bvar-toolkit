%% ex05 - A BVAR with common stochastic volatility
%
% BOOK: Chapter 14, Large VARs with Stochastic Volatility, in Bayesian
% Macroeconometrics: Methods and Applications (Chapman & Hall/CRC, forthcoming).
%
% THE MODEL. One log-volatility path scales the whole covariance matrix:
%
%       y_t' = x_t'A + u_t',   u_t ~ N(0, exp(h_t)*Sig),
%       h_t = phi*h_{t-1} + eps_t,   eps_t ~ N(0, sigh2),
%
% with h_1 ~ N(0, sigh2/(1-phi^2)). The mean of h is fixed at zero, so Sig
% carries the scale of the errors. This is the common stochastic volatility of
% Carriero, Clark and Marcellino (2016), implemented in Chan (2023): the
% sampler of replications/chan2023_joe_mlvarsv/legacy/VAR_CSV.m without its
% shrinkage-hyperparameter block.
%
% THE SAMPLER. bvar.models.var_csv, three blocks per sweep:
%
%   1. (A, Sig) | h          the natural-conjugate posterior of ex03, with each
%                            observation weighted by exp(-h_t)
%   2. h | A, Sig            bvar.sv.csv_armh
%   3. (phi, sigh2) | h      bvar.sv.sv0_params
%
% Given h the model is a weighted VAR, so block 1 draws all kn coefficients and
% Sig together from one k x k factorization, against the n factorizations of the
% equation-by-equation sampler in ex04.
%
% bvar.sv.csv_armh draws the whole path in one accept-reject Metropolis step, the
% algorithm of Chan (2020): Newton-Raphson for the mode of the conditional
% density, a Gaussian proposal there, an accept-reject screen with envelope
% constant c, then the Metropolis correction. The second section of the output
% runs the chain at three values of c, and the third estimates the model on data
% whose four equations have their own volatility paths.
%
% DATA. Simulated, so every path and parameter has a known truth.
%
% See:
% Carriero, A., Clark, T.E. and Marcellino, M. (2016). Common Drifting Volatility
% in Large Bayesian VARs, Journal of Business and Economic Statistics, 34(3):
% 375-390.
% Chan, J.C.C. (2020). Large Bayesian VARs: A flexible Kronecker error covariance
% structure, Journal of Business and Economic Statistics, 38(1): 68-79.
% Chan, J.C.C. (2023). Comparing Stochastic Volatility Specifications for Large
% Bayesian VARs, Journal of Econometrics, 235(2): 1419-1446.

run(fullfile(fileparts(fileparts(mfilename('fullpath'))),'setup.m'))

rng(20260917, 'twister')
fprintf('\n=== ex05: a BVAR with common stochastic volatility ===\n');

%% ------------------------------------------------------------------
%  1. Simulate from the model
%  ------------------------------------------------------------------
n = 4; p = 2; T = 500;
k = 1 + n*p;

A1_true = [ .5  .1   0   0
             0  .5  .1   0
            .1   0  .4   0
             0  .1   0  .4];
A2_true = .15*eye(n);
A_true  = [zeros(1,n); A1_true'; A2_true'];        % k x n, intercept first

sd_true  = [.6 .5 .4 .3];
R_true   = [ 1  .3  .2   0
            .3   1  .3  .1
            .2  .3   1  .2
              0  .1  .2   1];
Sig_true = diag(sd_true)*R_true*diag(sd_true);
CSig_true = chol(Sig_true, 'lower');

phi_true = .98; sigh2_true = .04;
Tall = T + 50;                                      % 50 rows to burn in the VAR
h_all = zeros(Tall,1);
h_all(1) = sqrt(sigh2_true/(1-phi_true^2))*randn;
for t = 2:Tall
    h_all(t) = phi_true*h_all(t-1) + sqrt(sigh2_true)*randn;
end
h_all = h_all - mean(h_all);                        % the model fixes the mean at zero

Yall = zeros(Tall, n);
for t = 3:Tall
    x = [1, Yall(t-1,:), Yall(t-2,:)];
    Yall(t,:) = x*A_true + exp(h_all(t)/2)*(CSig_true*randn(n,1))';
end
Y0 = Yall(Tall-T-3:Tall-T, :);                      % 4 presample rows, as niw wants
Y  = Yall(Tall-T+1:end, :);
h_true = h_all(Tall-T+1:end);
h_true = h_true - mean(h_true);

fprintf('\nsimulated n = %d, p = %d, T = %d\n', n, p, T);
fprintf('  true phi %.2f, sigh2 %.3f; h runs over [%.2f, %.2f]\n', ...
    phi_true, sigh2_true, min(h_true), max(h_true));
fprintf('  so the common volatility factor exp(h/2) runs over [%.2f, %.2f]\n', ...
    exp(min(h_true)/2), exp(max(h_true)/2));
fprintf('  series standard deviations %s, average correlation %.2f\n', ...
    mat2str(sd_true), mean(R_true(triu(true(n),1))));

%% ------------------------------------------------------------------
%  2. Estimate
%  ------------------------------------------------------------------
nsim = 2000; burnin = 500;
fprintf('\n1. ESTIMATION\n');
fprintf('running %d sweeps (%d kept)...\n', nsim+burnin, nsim);
t_est = tic;
est = bvar.models.var_csv(Y0, Y, p, 'nsim', nsim, 'burnin', burnin, ...
    'seed', 20260917, 'c_reject', 3, 'draws', true);
est.elapsed = toc(t_est);

h_band = quantile(est.draws.h, [.16 .84]);
cover  = mean(h_true' >= h_band(1,:) & h_true' <= h_band(2,:));
fprintf('done in %.1f seconds (%.1f ms per sweep)\n', est.elapsed, ...
    1e3*est.elapsed/(nsim+burnin));
fprintf('\n  %-34s %10s %10s\n', '', 'true', 'post mean');
fprintf('  %-34s %10.3f %10.3f\n', 'phi', phi_true, est.phi_mean);
fprintf('  %-34s %10.3f %10.3f\n', 'sigh2', sigh2_true, est.sigh2_mean);
fprintf('  %-34s %10.3f %10.3f\n', 'largest error sd, sqrt(Sig(1,1))', ...
    sqrt(Sig_true(1,1)), sqrt(est.Sig_mean(1,1)));
fprintf('\n  correlation of the posterior mean of h with the truth  %.3f\n', ...
    corr(est.h_mean, h_true));
fprintf('  68%% band covers the true h at %.0f%% of dates\n', 100*cover);
fprintf('  RMSE of the coefficient matrix A                       %.4f\n', ...
    sqrt(mean((est.A_mean(:) - A_true(:)).^2)));
fprintf('  accept-reject Metropolis step accepted %.0f%% of sweeps\n', ...
    100*est.accept_rate);
fprintf('  (the first 20 sweeps take the proposal regardless, to move h away\n');
fprintf('   from its starting path)\n');

%% ------------------------------------------------------------------
%  3. The screening constant of the accept-reject step
%  ------------------------------------------------------------------
    % the standardized residual sums of the last sweep, which the h draw
    % conditions on
[~, Xdes] = bvar.util.build_lags([Y0(end-p+1:end,:); Y], p);
A_last = reshape(est.draws.A(end,:), 1+n*p, n);
Sig_last = reshape(est.draws.Sig(end,:), n, n);
s2_last = sum(((Y - Xdes*A_last)/chol(Sig_last,'lower')').^2, 2);

fprintf('\n2. THE ENVELOPE CONSTANT c_reject\n');
fprintf('  The screen keeps a proposal with probability f(h)/(c*q(h)), so a\n');
fprintf('  larger c asks for more proposals and returns a candidate closer to\n');
fprintf('  the target, which the Metropolis step then rejects less often. c is\n');
fprintf('  not part of the target: every row below has the same posterior.\n\n');
c_list = [1.5 3 10];
short = cell(numel(c_list), 1);
bench = zeros(numel(c_list), 2);
for ic = 1:numel(c_list)
    short{ic} = bvar.models.var_csv(Y0, Y, p, 'nsim', 400, 'burnin', 100, ...
        'seed', 424242, 'c_reject', c_list(ic));
        % the cost of the h draw alone, at one fixed set of conditioning values:
        % the coefficients and covariance matrix of the last sweep above
    rng(11, 'twister');
    n_acc = 0; tic;
    for rep = 1:500
        [~, is_accept] = bvar.sv.csv_armh(s2_last, est.phi_mean, ...
            est.sigh2_mean, est.h_mean, n, false, [], 'c_reject', c_list(ic));
        n_acc = n_acc + is_accept;
    end
    bench(ic,:) = [1e3*toc/500, n_acc/500];
end
i_ref = find(c_list == 3);
fprintf('  %8s %16s %16s %24s\n', 'c', 'Metropolis kept', 'ms per h draw', ...
    'corr with the c = 3 run');
for ic = 1:numel(c_list)
    fprintf('  %8.1f %15.0f%% %16.2f %24.4f\n', c_list(ic), 100*bench(ic,2), ...
        bench(ic,1), corr(short{ic}.h_mean, short{i_ref}.h_mean));
end

%% ------------------------------------------------------------------
%  4. Four volatility paths, one common factor
%  ------------------------------------------------------------------
fprintf('\n3. DATA WITH FOUR SEPARATE VOLATILITY PATHS\n');
rng(20260918, 'twister')
phis = [.99 .97 .95 .98];
sigh2s = [.02 .04 .03 .02];
H_all = zeros(Tall, n);
for ii = 1:n
    H_all(1,ii) = sqrt(sigh2s(ii)/(1-phis(ii)^2))*randn;
    for t = 2:Tall
        H_all(t,ii) = phis(ii)*H_all(t-1,ii) + sqrt(sigh2s(ii))*randn;
    end
    H_all(:,ii) = H_all(:,ii) - mean(H_all(:,ii));
end
Yall2 = zeros(Tall, n);
for t = 3:Tall
    x = [1, Yall2(t-1,:), Yall2(t-2,:)];
    Yall2(t,:) = x*A_true + (diag(exp(H_all(t,:)/2))*CSig_true*randn(n,1))';
end
Y0b = Yall2(Tall-T-3:Tall-T, :);
Yb  = Yall2(Tall-T+1:end, :);
H_true = H_all(Tall-T+1:end, :);
H_true = H_true - mean(H_true);

estb = bvar.models.var_csv(Y0b, Yb, p, 'nsim', 1000, 'burnin', 250, ...
    'seed', 20260918, 'c_reject', 3);
fprintf('  the four true paths move apart: pairwise correlations %.2f to %.2f\n', ...
    min(nonzeros(tril(corr(H_true),-1))), max(nonzeros(tril(corr(H_true),-1))));
fprintf('\n  correlation of the single estimated path with\n');
for ii = 1:n
    fprintf('    the true path of equation %d   %6.3f\n', ii, corr(estb.h_mean, H_true(:,ii)));
end
fprintf('    their average                 %6.3f\n', corr(estb.h_mean, mean(H_true,2)));
fprintf('\n  One path can only follow the average. ex07 gives each series its own\n');
fprintf('  volatility through a factor structure, and ex04 gives each equation\n');
fprintf('  its own path.\n');

%% ------------------------------------------------------------------
%  5. Figure
%  ------------------------------------------------------------------
figure('Name','ex05: common stochastic volatility');
subplot(2,1,1);
hold on
bvar.util.shaded_band((1:T)', h_band(1,:)', h_band(2,:)', .85);
plot(1:T, est.h_mean, 'b-', 'LineWidth', 1.2);
plot(1:T, h_true, 'k--', 'LineWidth', 1);
hold off
box off; xlim([1 T]);
title('Common log-volatility h_t');
legend({'68% credible band','posterior mean','truth'}, 'Location','northwest', 'Box','off');

subplot(2,1,2);
hold on
hsep = plot(1:T, H_true, 'Color', [.7 .7 .7], 'LineWidth', .75);
set(hsep(2:end), 'HandleVisibility', 'off');
havg = plot(1:T, mean(H_true,2), 'k--', 'LineWidth', 1);
hcom = plot(1:T, estb.h_mean, 'b-', 'LineWidth', 1.2);
hold off
box off; xlim([1 T]);
title('Four separate paths, their average, and the estimated common path');
legend([hsep(1) havg hcom], {'the four true paths','their average','estimated common path'}, ...
    'Location','northwest', 'Box','off');

fprintf('\nex05 done. Next: ex06_variable_ordering_sv (does the order matter?).\n');

%% ------------------------------------------------------------------
