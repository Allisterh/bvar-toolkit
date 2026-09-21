%% build - regenerate the figures and numbers of tutorials/sv_specification/README.md
%
% The data are the five quarterly series of macro5_Q.csv in this folder: the four of
% the book's chapter12/macro4_Q.csv and the NFCI column of its
% chapter04/GDP_NFCI_merged.csv, joined on the quarter. The models, priors,
% lag length and marginal likelihood estimators are those of Chan (2023), archived
% in replications/chan2023_joe_mlvarsv. Part 1 estimates the seven specifications
% and their log marginal likelihoods with that package's run_ml, each under two
% seeds, and saves a summary of every run in runs/ next to this file; a run whose
% summary exists with the same settings is not repeated, so an interrupted build
% resumes where it stopped. Part 2 reports the marginal likelihoods, the shrinkage
% hyperparameters and the outlier probabilities, and Part 3 the MCMC diagnostics
% and the repeat estimates. Everything
% printed goes to a log in tempdir and the figures are written next to this file.
%
% Usage, from anywhere:  run tutorials/sv_specification/build.m
%
% The runs are independent. To spread them over several MATLAB sessions, set
% build_runs to a subset of 1:7 before running the script in each; the script then
% computes and saves only those runs. A final run without build_runs writes the log.

tdir = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(tdir));
only = exist('build_runs', 'var');
if ~only
    logf = fullfile(tempdir, 'bvar_sv_specification_build_log.txt');
    if exist(logf, 'file'), delete(logf); end
    diary(logf);
end
fprintf('tutorials/sv_specification/build.m, %s, MATLAB %s\n', ...
    char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm')), version);
t0 = tic;
run(fullfile(repo, 'setup.m'));
pkg = fullfile(repo, 'replications', 'chan2023_joe_mlvarsv');
addpath(pkg);

    % the five series as the book uses them: growth rates and inflation annualized,
    % the two rates and the NFCI in levels, no further scaling. The order is the
    % normalization of the Cholesky and factor specifications: unemployment leads,
    % so the first factor is normalized on a real variable, inflation follows for a
    % nominal one, and GDP growth, whose pandemic quarters are the most extreme,
    % comes last
vars = {'UNRATE','PCECTPI','FEDFUNDS','NFCI','GDPC1'};
raw = readtable(fullfile(tdir, 'macro5_Q.csv'), 'VariableNamingRule', 'preserve');
Q = raw{:, vars};
qd = raw{:, 'Date'};
if ~isdatetime(qd), qd = datetime(qd); end
assert(~any(isnan(Q(:))), 'the panel has missing values');
gap = days(diff(qd));
assert(all(gap >= 89 & gap <= 93), 'the quarters are not consecutive');
n0 = 8;  n = numel(vars);  T = size(Q, 1) - n0;
fprintf('data: %d quarters, %s to %s\n', size(Q, 1), qlab(qd(1)), qlab(qd(end)));
fprintf('estimation sample %s to %s, T = %d after %d initial conditions, n = %d, p = 4\n', ...
    qlab(qd(n0+1)), qlab(qd(end)), T, n0, n);

%% ------------------------------------------------------------------
%  Part 1. The seven specifications
%  ------------------------------------------------------------------
nsim = 20000;  burnin = 1000;  M = 10000;         % Section 5 of Chan (2023)
spec = struct('model', {'VAR-NCP', 'VAR-CSV', 'VAR-SV', 'VAR-SV', 'VAR-FSV', 'VAR-FSV', ...
                        'VAR-SVO'}, ...
              'sym',   {false, false, false, true, false, false, false}, ...
              'r',     {1, 1, 1, 1, 1, 2, 1});
nrun = numel(spec);
seeds = [20260919, 8177];                         % the repeat run refits chain, density and weights
key = [nsim, burnin, M, sum(Q(:)), 0];            % a saved run is reused only under these
rdir = fullfile(tdir, 'runs');
if ~exist(rdir, 'dir'), mkdir(rdir); end
S = cell(1, nrun);  R = cell(1, nrun);            % the two seeds
for i = 1:nrun
    S{i} = cached(rdir, i, 1, key);
    R{i} = cached(rdir, i, 2, key);
end
todo = find(cellfun(@isempty, S) | cellfun(@isempty, R));
fprintf('\n=== Part 1: %d draws after %d burn-in, %d importance-sampling draws, two seeds ===\n', ...
    nsim, burnin, M);
fprintf('%d of %d specifications saved by an earlier build\n', nrun - numel(todo), nrun);
if only, todo = intersect(todo, build_runs, 'stable'); end
for i = reshape(todo, 1, [])              % a 0-by-1 empty would run the body once
    if isempty(S{i}), S{i} = one_run(spec(i), Q, nsim, burnin, M, seeds(1) + i, key, rdir, i, 1); end
    if isempty(R{i}), R{i} = one_run(spec(i), Q, nsim, burnin, M, seeds(2) + i, key, rdir, i, 2); end
    fprintf('run %d, %s: %.1f and %.1f minutes\n', i, S{i}.name, S{i}.seconds/60, R{i}.seconds/60);
end
if only
    rmpath(pkg);
    return
end
fprintf('run times in minutes, both seeds:');
fprintf(' %d %.1f;', [1:nrun; (cellfun(@(s) s.seconds, S) + cellfun(@(s) s.seconds, R))/60]);
fprintf('\n');

%% ------------------------------------------------------------------
%  Part 2. Marginal likelihoods, shrinkage and outliers
%  ------------------------------------------------------------------
fprintf('\n=== Part 2: log marginal likelihoods ===\n');
lml = cellfun(@(s) s.lml, S);
nse = cellfun(@(s) s.nse, S);
main = [1 2 3 5 6 7];                             % Table 1; run 4 is the symmetric prior
[~, ib] = max(lml(main));  ib = main(ib);
fprintf('\nTable 1\n');
fprintf('| Model | log marginal likelihood | numerical standard error | difference from %s |\n', S{ib}.name);
fprintf('|---|---|---|---|\n');
for i = main
    fprintf('| %s | %.1f | %s | %.1f |\n', S{i}.name, lml(i), nsestr(nse(i)), lml(i) - lml(ib));
end
fprintf('VAR-NCP has no standard error: its log marginal likelihood is available in closed form\n');

fprintf('\nTable 2: the prior (posterior mean and standard deviation of each shrinkage hyperparameter)\n');
fprintf('| Model | log marginal likelihood | own lags | other lags | impact matrix |\n');
fprintf('|---|---|---|---|---|\n');
for i = [2 4 3 5 6 7]
    k = S{i}.kappa;
    c = repmat({''}, 1, 3);
    if size(k, 2) == 1                                    % VAR-CSV: one kappa for all lags
        c(1:2) = {msd(k)};
    else
        c(1:2) = {msd(k(:,1)), msd(k(:,2))};
        if size(k, 2) == 3, c{3} = msd(k(:,3)); end
    end
    fprintf('| %s | %.1f | %s | %s | %s |\n', S{i}.name, lml(i), c{:});
end
fprintf('symmetric minus asymmetric prior, VAR-SV: %.1f; VAR-SV (symmetric) minus VAR-CSV: %.1f\n', ...
    lml(4) - lml(3), lml(4) - lml(2));

fprintf('\n=== Part 2b: the outlier component of VAR-SVO ===\n');
so = S{7};
dq = qd(n0+1:end);
fprintf('posterior mean of the outlier probability po: %.4f (sd %.4f); prior mean %.4f\n', ...
    mean(so.po), std(so.po), 2.5/40);
fprintf('expected number of outlier quarters: %.1f of %d\n', sum(so.pout), T);
fprintf('quarters with an outlier probability above 0.5:\n');
for t = find(so.pout' > 0.5)
    fprintf('  %s: probability %.2f, posterior mean of o_t %.2f\n', qlab(dq(t)), so.pout(t), so.o_mean(t));
end
sc = S{2};
fprintf('VAR-CSV: posterior mean of exp(h_t/2) from %.2f (%s) to %.2f (%s)\n', ...
    min(sc.csv_std), qlab(dq(find(sc.csv_std == min(sc.csv_std), 1))), ...
    max(sc.csv_std), qlab(dq(find(sc.csv_std == max(sc.csv_std), 1))));

fig = figure('Visible', 'off', 'Units', 'centimeters', 'Position', [2 2 18 12]);
tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile; hold on; box off
plot(dq, sc.csv_std, 'k-', 'LineWidth', 1.2);
ylabel('e^{h_t/2}');
title('Common standard-deviation multiplier, VAR-CSV (posterior mean)');
nexttile; hold on; box off
bar(dq, so.pout, 1, 'FaceColor', [0.4 0.4 0.4], 'EdgeColor', 'none');
ylim([0 1.2]); ylabel('probability');
big = find(so.pout > 0.5);                        % the quarters the component marks
if ~isempty(big)
    text(dq(big(end)), 1.12, [strjoin(arrayfun(@(t) qlab(dq(t)), big(:)', ...
        'UniformOutput', false), ', ') ' '], 'HorizontalAlignment', 'right', 'FontSize', 9);
end
title('Posterior outlier probability under VAR-SVO');
exportgraphics(fig, fullfile(tdir, 'fig_outliers.png'), 'Resolution', 150);
close(fig)

%% ------------------------------------------------------------------
%  Part 3. MCMC diagnostics
%  ------------------------------------------------------------------
fprintf('\n=== Part 3: MCMC diagnostics (bvar.diag) ===\n');
L = 200;
fprintf('inefficiency factors at L = %d by parameter group, and the largest gap between the\n', L);
fprintf('posterior means of the two runs over all hyperparameters, in posterior sd\n');
fprintf('| Model | kappa | mu | phi | sigma^2 | largest gap between runs |\n');
fprintf('|---|---|---|---|---|---|\n');
gname = ["kappa" "mu" "phi" "sigma^2"];
drift = {[], [], [], []};              % |late - early| mean, in posterior sd, by group
for i = [2 3 4 5 6 7]
    s = S{i};  r = R{i};
    nk = size(s.kappa, 2);  m = size(s.hpara, 2);
    if m == 2                                   % VAR-CSV: [phi sig2], no mu
        grp = {s.kappa, [], s.hpara(:,1), s.hpara(:,2)};
        gid = [ones(1, nk), 3, 4];
    else
        q = m/3;
        grp = {s.kappa, s.hpara(:,1:q), s.hpara(:,q+1:2*q), s.hpara(:,2*q+1:3*q)};
        gid = [ones(1, nk), 2*ones(1, q), 3*ones(1, q), 4*ones(1, q)];
    end
    cells = strings(1, 4);
    for g = 1:4
        if isempty(grp{g}), cells(g) = "-";  continue; end
        IF = bvar.diag.inefficiency_factor(grp{g}, L);
        cells(g) = sprintf('%.0f to %.0f', min(IF), max(IF));
    end
    X1 = [s.kappa, s.hpara];  X2 = [r.kappa, r.hpara];
    sd = std([X1; X2]);
    fprintf('| %s | %s | %s | %s | %s | %.2f |\n', s.name, cells, ...
        max(abs(mean(X1) - mean(X2))./sd));
    N = size(X1, 1);  e = 1:round(.1*N);  l = round(.5*N)+1:N;
    d = abs([(mean(X1(l,:)) - mean(X1(e,:)))./sd; (mean(X2(l,:)) - mean(X2(e,:)))./sd]);
    for g = 1:4, drift{g} = [drift{g}, reshape(d(:, gid == g), 1, [])]; end
end
fprintf('\ndrift within a run, between the first tenth and the last half, in posterior sd,\n');
fprintf('over both runs of the six specifications\n');
for g = 1:4
    fprintf('  %-8s median %.2f, largest %.2f\n', gname(g), median(drift{g}), max(drift{g}));
end
fprintf('numerical standard errors of the log marginal likelihoods: %.2f to %.2f\n', min(nse(2:end)), max(nse(2:end)));

fprintf('\nrepeat runs: each specification estimated again from a second seed, which refits the\n');
fprintf('chain, the importance density and the weights\n');
fprintf('| Model | log ML, seed 1 | log ML, seed 2 | difference | standard error |\n');
fprintf('|---|---|---|---|---|\n');
for i = 1:nrun
    fprintf('| %s | %.1f | %.1f | %.1f | %s |\n', S{i}.name, S{i}.lml, R{i}.lml, ...
        R{i}.lml - S{i}.lml, nsestr(S{i}.nse));
end
    % what the normalization costs when the most extreme series leads: the same
    % one-factor model and seeds, with GDP growth moved to the front
alt = [5 1 2 3 4];
keyalt = [key(1:4), 1];
A = cell(1, 2);
for q = 1:2
    A{q} = cached(rdir, nrun + 1, q, keyalt);
    if isempty(A{q})
        A{q} = one_run(spec(5), Q(:, alt), nsim, burnin, M, seeds(q) + 5, keyalt, rdir, nrun + 1, q);
    end
end
fprintf('\nVAR-FSV, r = 1, with %s first instead of %s: log ML %.1f and %.1f from the two seeds,\n', ...
    vars{alt(1)}, vars{1}, A{1}.lml, A{2}.lml);
fprintf('%.1f apart, against %.1f in the order above\n', ...
    abs(A{2}.lml - A{1}.lml), abs(R{5}.lml - S{5}.lml));

fprintf('\nbuild finished in %.1f minutes (runs included: %.1f hours)\n', toc(t0)/60, ...
    (sum(cellfun(@(s) s.seconds, S)) + sum(cellfun(@(s) s.seconds, R)) + ...
     sum(cellfun(@(s) s.seconds, A)))/3600);
diary off
rmpath(pkg);

function s = cached(rdir, i, q, key)
% the saved run, when its settings match
s = [];
f = fullfile(rdir, sprintf('run%d_%d.mat', i, q));
if exist(f, 'file')
    c = load(f);
    if isequal(c.s.key, key), s = c.s; end
end
end

function s = one_run(sp, Q, nsim, burnin, M, seed, key, rdir, i, q)
% estimate one specification and its log marginal likelihood, and save what
% Parts 2 and 3 need; the draws of the VAR coefficients and log-volatilities are
% not kept
t0 = tic;
est = @() run_ml(sp.model, false, sp.sym, nsim, burnin, seed, [], 'data', Q, 'r', sp.r, 'M', M); %#ok<NASGU>
[~, out] = evalc('est()');                        % keeps the progress messages out of the log
s = struct('key', key, 'seed', seed, 'lml', out.lml, 'nse', NaN, 'seconds', []);
s.name = string(sp.model);
if sp.sym, s.name = s.name + " (symmetric prior)"; end
if strcmp(sp.model, 'VAR-FSV'), s.name = s.name + sprintf(", r = %d", sp.r); end
if ~isempty(out.lmlstd), s.nse = out.lmlstd; end
if isfield(out, 'store_kappa')
    s.kappa = out.store_kappa;  s.hpara = out.store_hpara;  s.count_phi = out.count_phi;
end
if isfield(out, 'CSV_std_mean'), s.csv_std = out.CSV_std_mean; end
if isfield(out, 'store_o')
    s.pout = mean(out.store_o > 1, 1)';  s.o_mean = out.o_mean;  s.po = out.store_po;
end
s.seconds = toc(t0);
save(fullfile(rdir, sprintf('run%d_%d.mat', i, q)), 's');
end

function s = msd(x)
s = sprintf('%.3g (%.2g)', mean(x), std(x));
end

function s = nsestr(x)
if isnan(x), s = '-'; else, s = sprintf('%.2f', x); end
end

function s = qlab(d)
s = sprintf('%dQ%d', year(d), quarter(d));
end

