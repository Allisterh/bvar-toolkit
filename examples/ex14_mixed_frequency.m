%% ex14 - A mixed-frequency VAR with common stochastic volatility
%
% THE MODEL. A monthly VAR(p) with the common stochastic volatility of Carriero, Clark and
% Marcellino (2016),
%
%       y_t = b_0 + B_1 y_{t-1} + ... + B_p y_{t-p} + e_t,   e_t ~ N(0, exp(h_t)*Sig),
%
% in which one series is observed only as a quarterly growth rate, the mixed-frequency VAR of
% Schorfheide and Song (2015). Its monthly growth rates are missing data, tied to each
% quarterly value z by the log-linear aggregation of Mariano and Murasawa (2003),
%
%       z = (y_t + 2*y_{t-1} + 3*y_{t-2} + 2*y_{t-3} + y_{t-4})/3,
%
% at the last month t of the quarter. bvar.util.mm_constraint stacks these into M*y = z.
%
% THE SAMPLER. bvar.models.mfvar_csv, the priors of Chan, Poon and Zhu (2023), five
% blocks per sweep: the missing values in one draw from their banded conditional, with
% M*y = z imposed exactly (bvar.samplers.missing_var); the coefficients jointly; Sig;
% the log-volatilities in blocks (bvar.sv.csv_armh_block); and their AR(1) parameters.
% The last two blocks follow the algorithm of Chan (2020), with the log-volatilities drawn
% in blocks.
%
% DATA. Generated, so the monthly values of the quarterly series are known. The output
% compares the posterior with them, and with the monthly path that sets each month of a
% quarter to a third of its quarterly growth.
%
% See:
% Chan, J.C.C., Poon, A. and Zhu, D. (2023). High-Dimensional Conditionally Gaussian
% State Space Models with Missing Data, Journal of Econometrics, 236(1): 105468.
% Carriero, A., Clark, T.E. and Marcellino, M. (2016). Common Drifting Volatility in
% Large Bayesian VARs, Journal of Business and Economic Statistics, 34(3): 375-390.
% Chan, J.C.C. (2020). Large Bayesian VARs: A Flexible Kronecker Error Covariance
% Structure, Journal of Business and Economic Statistics, 38(1): 68-79.
% Mariano, R.S. and Murasawa, Y. (2003). A New Coincident Index of Business Cycles
% Based on Monthly and Quarterly Series, Journal of Applied Econometrics, 18(4):
% 427-443.
% Schorfheide, F. and Song, D. (2015). Real-Time Forecasting with a Mixed-Frequency VAR,
% Journal of Business and Economic Statistics, 33(3): 366-380.

run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'setup.m'))
fprintf('\n=== ex14: a mixed-frequency VAR with common stochastic volatility ===\n');

%% generated data: two monthly indicators and one quarterly series
rng(14, 'twister');
n = 3; p = 2; T = 362;                           % 30 years of months after p initial ones
A = [0.1 0.1 0.2; 0.4 0.1 0.1; 0.1 0.3 0.1; 0.3 0.3 0.5; 0.1 0 0; 0 0.1 0; 0 0 0.1];
CS = [0.6 0 0; 0.2 0.5 0; 0.3 0.3 0.4];
phi = 0.98;  sigh = 0.05;
h = zeros(T,1);  h(1) = sigh/sqrt(1-phi^2)*randn;
for t = 2:T, h(t) = phi*h(t-1) + sigh*randn; end
Ytrue = zeros(T, n);
for t = p+1:T
    x = [1, reshape(Ytrue(t-1:-1:t-p,:)', 1, n*p)];
    Ytrue(t,:) = x*A + exp(h(t)/2)*randn(1,n)*CS';
end

% series 3 is published only as its quarterly growth, in the last month of each quarter
qend = (5:3:T)';
y3 = Ytrue(:,3);
X = Ytrue;
X(:,3) = NaN;
X(qend,3) = y3(qend - (0:4))*[1 2 3 2 1]'/3;
[M, z, Y] = bvar.util.mm_constraint(X, [false false true]);
fprintf('%d months, %d quarterly values of series 3, %d missing monthly values\n', ...
    T, numel(z), sum(isnan(Y(:))));

% the Minnesota scale of series 3: 9/19 of the AR(4) residual variance of its quarterly
% growth, the monthly variance that gives that quarterly variance under the aggregation
g = X(qend,3);
Zq = [ones(numel(g)-4,1) g(4:end-1) g(3:end-2) g(2:end-3) g(1:end-4)];
eq = g(5:end) - Zq*(Zq\g(5:end));
sig2 = [NaN NaN 9/19*mean(eq.^2)];

%% estimation
t0 = tic;
res = bvar.models.mfvar_csv(Y, p, 'M', M, 'z', z, 'sig2', sig2, ...
    'nsim', 4000, 'burnin', 1000, 'seed', 1);
fprintf('5,000 sweeps in %.1f seconds; share of h blocks accepted %.2f\n', ...
    toc(t0), res.accept_rate);
fprintf('M*y - z over the kept draws: max abs value %.2e\n', res.max_resid);

%% the monthly values of series 3 against the truth
i = p+1:T;
truth = Ytrue(i,3);
est = res.Y_mean(i,3);
band = squeeze(res.Y_q(i,3,[1 5]));
naive = nan(T,1);
for iq = 1:numel(qend)
    naive(qend(iq)-2:qend(iq)) = X(qend(iq),3)/3;
end
ok = ~isnan(naive(i));
fprintf('\nmonthly growth of series 3, %d months     RMSE   corr with truth\n', numel(i));
fprintf('posterior mean                         %7.3f  %7.3f\n', ...
    sqrt(mean((est - truth).^2)), corr(est, truth));
fprintf('a third of the quarterly growth        %7.3f  %7.3f   (%d months)\n', ...
    sqrt(mean((naive(i(ok)) - truth(ok)).^2)), corr(naive(i(ok)), truth(ok)), sum(ok));
fprintf('90%% credible bands: %.1f%% of the true values inside, mean width %.2f\n', ...
    100*mean(truth >= band(:,1) & truth <= band(:,2)), mean(band(:,2) - band(:,1)));

fprintf('\n                       true  posterior mean\n');
fprintf('phi                  %6.3f  %6.3f\n', phi, res.phi_mean);
fprintf('sigh2                %6.3f  %6.3f\n', sigh^2, res.sigh2_mean);
fprintf('corr(h, posterior mean of h) %.2f\n', corr(h(p+1:end), res.h_mean));

% the last five years: the true monthly values, the posterior mean and the 90% band
k = T-59:T;
figure; hold on; box off
bvar.util.shaded_band(k, res.Y_q(k,3,1), res.Y_q(k,3,5));
plot(k, Ytrue(k,3), 'r--', k, res.Y_mean(k,3), 'k', 'LineWidth', 1.2);
legend({'90% credible band', 'true monthly value', 'posterior mean'}, 'Location', 'best');
title('Series 3, observed only as quarterly growth: its monthly values');
xlabel('month'); hold off
