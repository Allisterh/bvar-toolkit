%% your_data - impose your own sign and ranking restrictions on a VAR
%
% Set the file, the columns, the lag length and the restrictions below, then run
% the script. It estimates a VAR under the asymmetric conjugate prior of Chan
% (2022), with the two shrinkage hyperparameters chosen by maximizing the
% marginal likelihood, draws candidate rotations until the requested number are
% admissible under bvar.structural.sign_assign, and reports the impulse
% responses with 68 per cent credible bands. The same candidates are also tested
% by the accept-reject rule of Rubio-Ramirez, Waggoner and Zha (2010), so the
% run reports what each rule accepts.
%
% Restrictions are given per shock as two index lists into cols: the variables
% whose impact response must be positive, and those whose must be negative.
% Every other response is free. A ranking restriction is a row [shock, a, b]
% meaning that for that shock the impact response of variable a is at most the
% impact response of variable b.
%
% The restrictions must satisfy Assumption 2 of Chan, Matthes and Yu (2026),
% which separates every pair of shocks either by their signs or by a ranking
% restriction. The script checks it before it draws.
%
% The defaults restrict three shocks in a five-variable VAR and take a few
% seconds. The tutorial itself restricts eight shocks in 35 variables.
%
% Give cols as column names, read with readtable. The selected columns must have
% no missing values inside the sample; rows missing at either end are dropped.

tdir = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(tdir));
run(fullfile(repo, 'setup.m'))

%% ---- settings ----
file    = fullfile(repo, 'tutorials', 'sv_specification', 'macro5_Q.csv');
cols    = ["GDPC1" "PCECTPI" "FEDFUNDS" "UNRATE" "NFCI"];
label   = ["GDP growth" "PCE inflation" "Fed funds" "Unemployment" "NFCI"];
datecol = "Date";                    % the column holding the dates; "" to skip
p       = 4;        % lags
n0      = 8;        % rows used as initial conditions
horizon = 20;       % horizons reported, counting the impact period
nkeep   = 100;      % admissible draws to collect
nbatch  = 500;      % posterior draws per batch, one candidate rotation each
levels  = [];       % columns whose first own lag gets prior mean one
ridge   = 0;        % added to the posterior precision in the marginal likelihood
seed    = 1;
outdir  = tempdir;  % where the report goes; '' for none

shock = ["demand" "supply" "monetary"];
pos = {[1 2 3], ...             % demand:   output, prices and the policy rate rise
       1, ...                   % supply:   output rises
       [3 4]};                  % monetary: the policy rate and unemployment rise
neg = {4, ...                   % demand:   unemployment falls
       [2 3 4], ...             % supply:   prices, the policy rate and unemployment fall
       [1 2]};                  % monetary: output and prices fall
ranking = [3 1 2];              % under shock 3, variable 1 rises by no more than
                                % variable 2: output falls by at least as much as
                                % prices under a monetary contraction

%% ---- data ----
raw = readtable(file, 'VariableNamingRule', 'preserve');
sel = raw{:, cols};
keep = all(isfinite(sel), 2);
assert(any(keep), 'every row of the selected columns has a missing value');
lo = find(keep, 1);  hi = find(keep, 1, 'last');
assert(all(keep(lo:hi)), 'the selected columns have missing values inside the sample');
data = sel(lo:hi, :);
[nobs, n] = size(data);
m = numel(shock);
assert(numel(pos) == m && numel(neg) == m, 'pos and neg need one entry per shock');
assert(numel(label) == n, 'label needs one entry per column');
Y0 = data(1:n0,:);  Y = data(n0+1:end,:);
[~, Z] = bvar.util.build_lags([Y0(end-p+1:end,:); Y], p);

if strlength(datecol) > 0 && isdatetime(raw{:, datecol})
    d = raw{lo:hi, datecol};
    fprintf('\n%d variables, %d observations (%s to %s), %d lags\n', n, nobs, ...
        string(d(1),'uuuuQQQ'), string(d(end),'uuuuQQQ'), p);
else
    fprintf('\n%d variables, %d observations, %d lags\n', n, nobs, p);
end

%% ---- the restrictions ----
S = nan(n, m);
for i = 1:m
    S(pos{i},i) = 1;
    S(neg{i},i) = -1;
end
kr = 1;
if ~isempty(ranking), kr = max(histcounts(ranking(:,1), .5:1:(m+.5))); end
Rineq = zeros(m, n, kr);
cnt = zeros(m, 1);
Rrows = zeros(0, n);  Ridx = zeros(0, 1);
for r = 1:size(ranking,1)
    s = ranking(r,1);  cnt(s) = cnt(s) + 1;
    Rineq(s, ranking(r,2), cnt(s)) =  1;
    Rineq(s, ranking(r,3), cnt(s)) = -1;
    Rrows(end+1,:) = Rineq(s,:,cnt(s));   %#ok<SAGROW>
    Ridx(end+1,1) = s;                    %#ok<SAGROW>
end
fprintf('%d sign restrictions and %d ranking restrictions over %d shocks\n', ...
    sum(~isnan(S(:))), size(Rrows,1), m);

% Every pair of shocks must be separated, or one column can serve two shocks
% and the assignment is not defined. sign_assign runs once per candidate and
% does not check it.
[sep, unsep] = bvar.structural.check_separable(S, Rineq);
if ~sep
    error(['shocks %d and %d are not separable: Assumption 2 of Chan, Matthes ' ...
        'and Yu (2026) needs either one variable restricted with the same sign ' ...
        'under both and one with opposite signs, or a ranking restriction on the ' ...
        'same two variables pointing opposite ways'], unsep(1,1), unsep(1,2));
end

%% ---- the prior, with its shrinkage chosen by marginal likelihood ----
[ml_opt, kappa] = bvar.priors.acp_opt_kappa(Y0, Y, Z, p, [.04 .0016], 'redu', ...
    levels, 'ridge', ridge);
sig2 = bvar.priors.resid_var_ar4(Y0, Y);
prior = bvar.priors.acp_redu(n, p, kappa, sig2, levels);
fprintf('shrinkage by marginal likelihood: kappa1 %.4g, kappa2 %.4g, log ML %.1f\n', ...
    kappa(1), kappa(2), ml_opt);

%% ---- draw until nkeep rotations are admissible ----
rng(seed, 'twister');
resp = zeros(nkeep, n, m, horizon);
nsat = 0;  ntotal = 0;  nstrict = 0;
t0 = tic;
while nsat < nkeep
    [alp, beta, sig] = bvar.samplers.acp_theta_sig(Y0, Y, p, prior, nbatch);
    [Btilde, Sigtilde] = bvar.structural.reduced_form(alp, beta, sig);
    for isim = 1:nbatch
        ntotal = ntotal + 1;
        L = chol(squeeze(Sigtilde(isim,:,:)), 'lower')*bvar.structural.qr_sign(randn(n));
        nstrict = nstrict + bvar.structural.sign_restrict(L, S, Rrows, Ridx);
        [ok, Lo] = bvar.structural.sign_assign(L, S, Rineq, kr);
        if ok && nsat < nkeep
            nsat = nsat + 1;
            B = reshape(Btilde(isim,:), n*p+1, n);
            resp(nsat,:,:,:) = bvar.structural.irf_redu(B(2:end,:), Lo, horizon, m);
        end
    end
end
fprintf('%d admissible draws from %d candidates, %.0f each, in %.1f seconds\n', ...
    nsat, ntotal, ntotal/nsat, toc(t0));
fprintf('the accept-reject rule accepted %d of the same candidates\n', nstrict);

%% ---- the impulse responses ----
lo68 = squeeze(quantile(resp, .16, 1));
md = squeeze(median(resp, 1));
hi68 = squeeze(quantile(resp, .84, 1));

fprintf('\nimpact response to a one-standard-deviation shock, posterior median\n');
fprintf('  %-18s', 'variable');  fprintf('%16s', shock);  fprintf('\n');
for i = 1:n
    fprintf('  %-18.18s', label(i));
    fprintf('%16.3f', md(i,:,1));
    fprintf('\n');
end

for is = 1:m
    figure('Position', [100 100 780 (1 + (n > 3))*180 + 90]);
    for i = 1:n
        subplot(1 + (n > 3), ceil(n/(1 + (n > 3))), i);
        hold on
        bvar.util.shaded_band((0:horizon-1)', squeeze(lo68(i,is,:)), ...
            squeeze(hi68(i,is,:)), .85);
        plot(0:horizon-1, squeeze(md(i,is,:)), 'k--', 'LineWidth', 1.1);
        yline(0, 'k:');
        hold off; box off; xlim([0 horizon-1]);
        title(label(i));
    end
    sgtitle(sprintf('%s shock', shock(is)));
end

%% ---- the report ----
if ~isempty(outdir)
    rows = n*m*horizon;
    T1 = table('Size', [rows 6], 'VariableTypes', ...
        {'string','string','double','double','double','double'}, ...
        'VariableNames', {'shock','variable','horizon','p16','median','p84'});
    r = 0;
    for is = 1:m
        for i = 1:n
            for h = 1:horizon
                r = r + 1;
                T1(r,:) = {shock(is), cols(i), h-1, lo68(i,is,h), md(i,is,h), hi68(i,is,h)};
            end
        end
    end
    meta = struct('file', file, 'columns', cols, 'p', p, 'n0', n0, 'n', n, ...
        'T', size(Y,1), 'shocks', shock, 'horizon', horizon, 'nkeep', nkeep, ...
        'candidates', ntotal, 'accepted_strict', nstrict, 'kappa', kappa, ...
        'log_ml', ml_opt, 'ridge', ridge, 'seed', seed);
    bvar.util.report('sign_restrictions', struct('irf', T1), meta, outdir);
end
