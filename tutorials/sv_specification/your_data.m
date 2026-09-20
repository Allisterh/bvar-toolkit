%% your_data - compare stochastic volatility specifications of a VAR on your data
%
% Set the file, the columns and the chain lengths below, then run the script. It
% estimates a homoskedastic VAR and four VARs with stochastic volatility: common
% (VAR-CSV), Cholesky (VAR-SV, under the asymmetric and the symmetric prior),
% factor (VAR-FSV, for each number of factors in rs) and Cholesky with an outlier
% component (VAR-SVO). It then prints the log marginal likelihood of each with its
% numerical standard error, plots the posterior probability that each period is an
% outlier, and writes the comparison to a csv and a mat file. The models, priors
% and estimators are those of Chan (2023): four lags, the first eight rows as
% initial conditions, and prior means of zero on the VAR coefficients, so the
% series must be stationary. The defaults use the five series of the tutorial with
% short chains; the tutorial keeps 20,000 draws after 1,000 burn-in and uses
% 10,000 importance-sampling draws.
%
% Give cols as column names, read with readtable. The selected columns must have
% no missing values inside the sample; rows missing at either end are dropped.
% With rows = "months" the file needs a date column in the book's convention
% (year + month/12), and the months of each quarter are averaged; otherwise the
% rows are used as they are, and a date column, when given, is checked for even
% spacing.

tdir = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(tdir));
run(fullfile(repo, 'setup.m'))
lastwarn('');       % the report records the last warning raised from here on

%% ---- settings ----
file    = fullfile(tdir, 'macro5_Q.csv');
cols    = ["UNRATE" "PCECTPI" "FEDFUNDS" "NFCI" "GDPC1"];    % the order is a normalization
pct     = [false false false false false];   % which columns to multiply by 100
datecol = "Date";                    % the column holding the dates; "" to skip the check
rows    = "periods";                 % "months": average the months of each quarter;
                                     % "periods": use the rows as they are
rs      = [1 2];                     % numbers of factors in VAR-FSV
nsim    = 1000;                      % draws kept
burnin  = 200;                       % draws discarded first
M       = 1000;                      % importance-sampling draws
seed    = 1;
outdir  = tempdir;                   % where the report goes; '' for none

%% ---- data ----
raw = readtable(file, 'VariableNamingRule', 'preserve');
sel = raw{:, cols};
sel(:, pct) = 100*sel(:, pct);
keep = all(isfinite(sel), 2);
assert(any(keep), 'every row of the selected columns has a missing value');
lo = find(keep, 1);  hi = find(keep, 1, 'last');
assert(all(keep(lo:hi)), 'the selected columns have missing values inside the sample');
data = sel(lo:hi, :);
flat = std(data) == 0;
assert(~any(flat), 'these series are constant: %s', strjoin(cols(flat), ', '));
dates = [];
if strlength(datecol) > 0, dates = raw{lo:hi, datecol}; end
lab = compose('%d', (1:size(data, 1))');
if rows == "months"
    assert(~isempty(dates), 'averaging months into quarters needs a date column');
    am = round(12*dates);                            % year*12 + month
    assert(all(diff(am) == 1), 'the months kept are not consecutive');
    yr = floor((am - 1)/12);
    [q, ~, iq] = unique(4*yr + ceil((am - 12*yr)/3));
    cnt = accumarray(iq, 1);
    avg = zeros(numel(q), numel(cols));
    for j = 1:numel(cols), avg(:, j) = accumarray(iq, data(:, j))./cnt; end
    data = avg(cnt == 3, :);
    q = q(cnt == 3);
    lab = compose('%dQ%d', floor((q - 1)/4), q - 4*floor((q - 1)/4));
elseif ~isempty(dates)
    if isdatetime(dates)
        step = days(diff(dates));
        even = all(step > 0) && max(abs(step - median(step))) <= 3;   % calendar months vary
    else
        step = diff(double(dates));
        even = all(step > 0) && max(abs(step - median(step))) <= 1e-6*max(1, abs(median(step)));
    end
    assert(even, 'the dates of the rows kept are not evenly spaced');
    if isdatetime(dates), lab = cellstr(string(dates, 'uuuuQQQ')); end
end
fprintf('\n%d variables, %d observations after 8 initial conditions (%s to %s)\n', ...
    numel(cols), size(data, 1) - 8, lab{9}, lab{end});

%% ---- the specifications and their marginal likelihoods ----
pkg = fullfile(repo, 'replications', 'chan2023_joe_mlvarsv');
addpath(pkg);
back = onCleanup(@() rmpath(pkg));
spec = [{'VAR-NCP', false, 1}; {'VAR-CSV', false, 1}; {'VAR-SV', false, 1}; {'VAR-SV', true, 1}; ...
        [repmat({'VAR-FSV'}, numel(rs), 1), repmat({false}, numel(rs), 1), num2cell(rs(:))]; ...
        {'VAR-SVO', false, 1}];
nspec = size(spec, 1);
label = strings(nspec, 1);
lml = zeros(nspec, 1);  nse = nan(nspec, 1);  secs = zeros(nspec, 1);
kap = cell(nspec, 1);  hyper = cell(nspec, 1);
for s = 1:nspec
    t0 = tic;
    res = run_ml(spec{s,1}, false, spec{s,2}, nsim, burnin, seed, [], 'data', data, ...
        'r', spec{s,3}, 'M', M);
    lml(s) = res.lml;
    if ~isempty(res.lmlstd), nse(s) = res.lmlstd; end
    label(s) = string(spec{s,1});
    if spec{s,2}, label(s) = label(s) + ", symmetric prior"; end
    if strcmp(spec{s,1}, 'VAR-FSV'), label(s) = label(s) + sprintf(", r = %d", spec{s,3}); end
    if strcmp(spec{s,1}, 'VAR-SVO'), pout = mean(res.store_o > 1, 1)'; omean = mean(res.store_o, 1)'; end
    if isfield(res, 'kappa_mean'), kap{s} = res.kappa_mean(:)'; end
    hyper{s} = res.Hyper;
    secs(s) = toc(t0);
    fprintf('%s: %.0f s\n', label(s), secs(s));
end
T = res.T;  n = res.n;        % the estimation sample, past the 8 initial conditions

fprintf('\nlog marginal likelihood (numerical standard error)\n');
[~, best] = max(lml);
for s = 1:nspec
    if isnan(nse(s)), se = "    -"; else, se = compose("%5.2f", nse(s)); end
    fprintf('%-32s %11.1f (%s) %9.1f\n', label(s), lml(s), se, lml(s) - lml(best));
end
fprintf('The last column is the difference from %s. VAR-NCP''s marginal likelihood is\n', label(best));
fprintf('available in closed form, so it has no standard error.\n');

fprintf('\nperiods with a posterior outlier probability above 0.5 in VAR-SVO:');
fprintf(' %s', lab{8 + find(pout > 0.5)});
fprintf('\n');
figure;
bar(pout, 1, 'FaceColor', [0.4 0.4 0.4], 'EdgeColor', 'none');
tk = round(linspace(1, numel(pout), 6));
set(gca, 'XTick', tk, 'XTickLabel', lab(8 + tk));
ylim([0 1]); box off
ylabel('posterior probability');
title('Probability that each period is an outlier, VAR-SVO');

%% ---- the report ----
% One row per specification, the outlier probabilities of VAR-SVO in their own
% file, and the settings in the mat file, which also holds the prior
% hyperparameters and the posterior mean of the shrinkage kappa of each model.
if ~isempty(outdir)
    rcol = cell2mat(spec(:,3));
    rcol(~strcmp(spec(:,1), 'VAR-FSV')) = NaN;    % r is the number of factors, FSV only
    models = table(label, cell2mat(spec(:,2)), rcol, lml, nse, ...
        lml - lml(best), secs, repmat([n T res.p nsim burnin M seed], nspec, 1), ...
        'VariableNames', {'model', 'symmetric_prior', 'r', 'log_ML', 'nse', ...
        'gap_to_best', 'seconds', 'settings'});
    models = splitvars(models, 'settings', 'NewVariableNames', ...
        {'n', 'T', 'p', 'nsim', 'burnin', 'M', 'seed'});
    outliers = table(string(lab(9:end)), pout, omean, ...
        'VariableNames', {'period', 'probability', 'mean_multiplier'});
    meta = struct('file', file, 'columns', cols, 'pct', pct, 'rows', rows, ...
        'datecol', datecol, 'n', n, 'T', T, 'p', res.p, 'rs', rs, 'nsim', nsim, ...
        'burnin', burnin, 'M', M, 'seed', seed, 'sample', string(lab([9 end])), ...
        'kappa_mean', {kap}, 'hyper', {hyper});
    bvar.util.report('sv_specification_report', ...
        struct('models', models, 'outliers', outliers), meta, outdir);
end
