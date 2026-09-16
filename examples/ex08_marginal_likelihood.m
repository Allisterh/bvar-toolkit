%% ex08 - Marginal likelihoods and model comparison
%
% BOOK: Chapter 5, Bayesian Model Comparison, in Bayesian Macroeconometrics:
% Methods and Applications (Chapman & Hall/CRC, forthcoming).
%
% THE QUANTITY. The marginal likelihood p(y | M), the integral of the likelihood
% against the prior, computed by Chib's method at theta*, the posterior means:
%
%       log p(y) = log p(y | theta*) + log p(theta*) - log p(theta* | y).
%
% The script compares three error specifications of Chan (2020, JBES), with the
% VAR and the prior held fixed: iid Gaussian errors (BVAR), Student-t errors
% (BVAR-t) and a common stochastic volatility factor (BVAR-CSV). It runs
% replications/chan2020_jbes_kronecker/run_ml.m for each model and prints the
% three terms for BVAR-t separately. The chains are a few hundred draws, against
% 30,000 in the paper.
%
% DATA. That package's quarterly US panel, data_Q.csv, read-only.
%
% See:
% Chan, J.C.C. (2020). Large Bayesian VARs: A flexible Kronecker error
% covariance structure, Journal of Business and Economic Statistics, 38(1), 68-79.

run(fullfile(fileparts(fileparts(mfilename('fullpath'))),'setup.m'))

repo = fileparts(fileparts(mfilename('fullpath')));
fprintf('\n=== ex08: marginal likelihoods and model comparison ===\n');

nsim = 200; burnin = 50; seed = 20260903;
fprintf('\nsettings: nsim = %d, burnin = %d, seed = %d\n', nsim, burnin, seed);
fprintf('(the published run uses nsim = 30000, burnin = 5000)\n');

%% ------------------------------------------------------------------
%  Run the driver from its own folder, so run_all, preset and the legacy
%  data file resolve unambiguously (three replication packages define a
%  function called run_all). The working directory is restored below
%  whatever happens.
%  ------------------------------------------------------------------
kdir  = fullfile(repo, 'replications', 'chan2020_jbes_kronecker');
oldwd = cd(kdir);
try
    fprintf('\n--- model 1: BVAR (iid Gaussian) ---\n');
    o1 = run_ml(1, nsim, burnin, seed);
    fprintf('\n--- model 2: BVAR-t (Student-t errors) ---\n');
    o2 = run_ml(2, nsim, burnin, seed);
    fprintf('\n--- model 3: BVAR-CSV (common stochastic volatility) ---\n');
    o3 = run_ml(3, nsim, burnin, seed);
catch err
    cd(oldwd);
    rethrow(err);
end
cd(oldwd);

%% ------------------------------------------------------------------
%  1. Where the number comes from: the three pieces of Chib's identity
%  ------------------------------------------------------------------
fprintf('\nlog p(y) = log p(y|theta*) + log p(theta*) - log p(theta*|y), for BVAR-t:\n');
fprintf('  log likelihood at theta*                      : %12.4f\n', o2.ml.llike);
fprintf('  log prior at theta*                           : %12.4f\n', o2.ml.lpri);
fprintf('  log posterior ordinate, (A,Sig) block         : %12.4f\n', o2.ml.lpost(1));
fprintf('  log posterior ordinate, nu block              : %12.4f\n', o2.ml.lpost(2));
fprintf('  ------------------------------------------------------------\n');
fprintf('  log marginal likelihood                       : %12.4f\n', o2.ML);
fprintf('  (posterior mean of the t degrees of freedom nu : %.2f - well below\n', o2.ml.nu_mean);
fprintf('   the ~30 at which a t is indistinguishable from a normal, so the\n');
fprintf('   posterior concentrates on fat tails.)\n');

fprintf('\nNote the two ordinate blocks are SUBTRACTED. A model can raise its\n');
fprintf('likelihood by fitting the sample more closely and still lose, because\n');
fprintf('a sharper posterior means a larger ordinate at theta*. That subtraction\n');
fprintf('is the Ockham factor: the marginal likelihood prices complexity\n');
fprintf('automatically, with no penalty term added to it.\n');

%% ------------------------------------------------------------------
%  2. The comparison
%  ------------------------------------------------------------------
names = {'BVAR      (iid Gaussian)', 'BVAR-t    (Student-t)', 'BVAR-CSV  (common SV)'};
MLs   = [o1.ML, o2.ML, o3.ML];
[~, best] = max(MLs);

fprintf('\n%s\n', repmat('-', 1, 66));
fprintf('%-26s %14s %12s %10s\n', 'model', 'log ML', 'log BF vs 1', 'rank');
fprintf('%s\n', repmat('-', 1, 66));
[~, ord] = sort(MLs, 'descend');
rank = zeros(1,3); rank(ord) = 1:3;
for ii = 1:3
    fprintf('%-26s %14.2f %12.2f %10d%s\n', names{ii}, MLs(ii), MLs(ii) - MLs(1), ...
        rank(ii), repmat('  <- best', 1, ii == best));
end
fprintf('%s\n', repmat('-', 1, 66));

fprintf('\nHow to read the middle column: it is a log Bayes factor against the\n');
fprintf('plain Gaussian VAR. On Kass and Raftery''s scale a log BF above 5 is\n');
fprintf('"very strong" evidence, and these are far larger - allowing for fat\n');
fprintf('tails or time-varying volatility improves the fit by a wide margin,\n');
fprintf('which is the paper''s main finding. The full-length ranking puts the\n');
fprintf('model that does BOTH (plus an MA term) on top; see tests/golden/ for\n');
fprintf('those tables.\n');

%% ------------------------------------------------------------------
%  3. Using the ml functions on your own model
%  ------------------------------------------------------------------
fprintf('\nTo compute an ML for a different model, the pattern is:\n');
fprintf('  out = run_ml(model, nsim, burnin, seed)   %% estimation + ML, one stream\n');
fprintf('  out.ML          the log marginal likelihood\n');
fprintf('  out.ml.llike / .lpri / .lpost    the three pieces above\n');
fprintf('  out.est         the draws, if you want them\n');
fprintf('The bvar.ml.* functions can also be called directly on stored draws;\n');
fprintf('their headers document the pri/est structs they expect.\n');

fprintf('\nOne footnote for replication: two of the eight legacy ML scripts\n');
fprintf('(models 4 and 8) evaluate one term at a leftover chain draw rather\n');
fprintf('than the posterior mean. The core functions use the posterior mean by\n');
fprintf('default and reproduce the published computation under\n');
fprintf('''bugcompat'', true. It does not affect the ranking; the audit and the\n');
fprintf('comparison are in tests/variant_map.md.\n');

fprintf('\nex08 done. For a published number use the full settings:\n');
fprintf('  cd replications/chan2020_jbes_kronecker\n');
fprintf('  out = run_ml(3, 30000, 5000);\n');
