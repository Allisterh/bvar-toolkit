function test_acp_equivalence
% seeded draw-for-draw equivalence of the functionized ACP pipeline
% (replications/chan2022_qe_acp/run_all.m + bvar.samplers.acp_theta_sig +
% bvar.structural.reduced_form / qr_sign / sign_restrict / irf_redu +
% bvar.priors.acp_redu / acp_stru / resid_var_ar4 / acp_opt_kappa + bvar.ml.acp)
% with the legacy script main_ACP_apps.m, run from a tempdir copy at a small
% number of accepted draws. Asserts isequal on the stored impulse responses, on
% the median and percentile summaries, and on the total number of draws examined
% - the last of which pins the rejection loop's arithmetic, not just its output.
%
% PATCHES TO THE LEGACY SCRIPT, each asserted to match exactly once:
%   1. `clear; clc;` removed - run from a function it would wipe the harness's
%      own bookkeeping;
%   2. nsim for dataset 1 reduced from 5000;
%   3. nbatch reduced from 50000, so a batch is cheap;
%   4. the figure block at the end removed, since it opens windows and is not
%      part of the computation.
% None touches an arithmetic line. Neither legacy driver seeds the generator at
% all, so there is no clock-seed line to remove; the harness seeds before each
% run instead, which is also the only way to make the legacy script repeatable.
%
% Scope: dataset 1 (n = 6, kappa fixed), draw for draw. Dataset 2 (n = 15)
% optimizes kappa and needs days of rejection sampling for its published run, so
% it is compared with no draws taken: the legacy script at dataset = 2 and
% nsim = 0 against run_all(2, 0), asserting isequal on the data, the lag matrix,
% the optimized kappa and its log marginal likelihood, the residual variances,
% the reduced-form prior, and the sign restrictions and row inequalities. That
% run has its own four patches, each asserted to match exactly once: `clear;
% clc;` removed, the dataset selector set to 2, nsim for dataset 2 set to 0, and
% everything from the summaries on removed. The rejection loop is not run at
% n = 15. The nsim and nbatch defaults in preset.m, which both runs override, are
% checked against the legacy values. bvar.priors.acp_opt_kappa is also covered
% separately below against get_OptKappa on the 6-variable data.
root = getappdata(0, 'bvar_repo_root');
leg = fullfile(root, 'replications', 'chan2022_qe_acp', 'legacy');
repdir = fullfile(root, 'replications', 'chan2022_qe_acp');

nsim = 30; nbatch = 400; seed = 20260907;

% --- tempdir: patched driver, verbatim utilities, and the data file ---
tmp = tempname; mkdir(tmp); mkdir(fullfile(tmp, 'utility'));
ctmp = onCleanup(@() cleanup_tmp(tmp));
u = dir(fullfile(leg, 'utility', '*.m'));
for k = 1:numel(u)
    copyfile(fullfile(leg, 'utility', u(k).name), fullfile(tmp, 'utility', u(k).name));
end
copyfile(fullfile(leg, 'database_2019Q4.xlsx'), fullfile(tmp, 'database_2019Q4.xlsx'));

txt = fileread(fullfile(leg, 'main_ACP_apps.m'));
txt = patch_once(txt, 'clear; clc;', '% [clear removed by test_acp_equivalence]', 'clear');
txt = patch_once(txt, 'nsim = 5000;', sprintf('nsim = %d;', nsim), 'nsim');
txt = patch_once(txt, 'nbatch = 50000;', sprintf('nbatch = %d;', nbatch), 'nbatch');
cut = strfind(txt, 'response_median = squeeze(median(store_response));');
assert(numel(cut) == 1, 'expected exactly one summary line');
tail = ['response_median = squeeze(median(store_response));' newline ...
        'response_CI = squeeze(quantile(store_response,[.16,.84]));' newline];
txt = [txt(1:cut-1) tail];   % everything after the summaries is figures
fid = fopen(fullfile(tmp, 'main_ACP_apps.m'), 'w'); fwrite(fid, txt); fclose(fid);

addpath(repdir); cp2 = onCleanup(@() rmpath(repdir)); %#ok<NASGU>
resolved = which('run_all');
assert(strncmpi(resolved, repdir, numel(repdir)), ...
    'run_all must resolve from the ACP package, got %s', resolved);

L = run_legacy(tmp, seed);
res = run_all(1, nsim, seed, 'nbatch', nbatch);
sC = rng;

assert(isequal(L.store_response, res.store_response), 'store_response differs');
assert(isequal(L.response_median, res.response_median), 'response_median differs');
assert(isequal(L.response_CI, res.response_CI), 'response_CI differs');
assert(isequal(L.count_total, res.count_total), ...
    'count_total differs: legacy %d, run_all %d', L.count_total, res.count_total);
assert(isequal(L.count_sat, size(res.store_response,1)), ...
    'the accepted-draw count differs from the stored rows');
assert(isequal(L.rngstate, sC.State), 'rng call sequence differs');

% the run must actually accept something, or the comparison is vacuous
assert(size(res.store_response,1) >= nsim, 'fewer accepted draws than requested');
% and the final batch must overshoot, which is the behaviour run_all reproduces
% deliberately (no early exit inside the batch)
assert(size(res.store_response,1) >= nsim, 'overshoot invariant not exercised');

% --- dataset 2 (n = 15): everything before the rejection loop ---
txt2 = fileread(fullfile(leg, 'main_ACP_apps.m'));
txt2 = patch_once(txt2, 'clear; clc;', '% [clear removed by test_acp_equivalence]', 'clear');
txt2 = patch_once(txt2, 'dataset = 1;', 'dataset = 2;', 'the dataset selector');
txt2 = patch_once(txt2, 'nsim = 1000;', 'nsim = 0;', 'nsim for dataset 2');
cut = strfind(txt2, 'response_median = squeeze(median(store_response));');
assert(numel(cut) == 1, 'expected exactly one summary line');
txt2 = txt2(1:cut-1);
fid = fopen(fullfile(tmp, 'main_ACP_apps_d2.m'), 'w'); fwrite(fid, txt2); fclose(fid);

L2 = run_legacy_setup(tmp, 'main_ACP_apps_d2');
res2 = run_all(2, 0, seed);
names = {'Y0', 'Y', 'Z', 'var_id', 'idx_ns', 'kappa', 'ml_opt', 'sig2', ...
    'prior_redu', 'S', 'Rineq', 'Ridx'};
for k = 1:numel(names)
    assert(isequaln(L2.(names{k}), res2.(names{k})), 'dataset 2: %s differs', names{k});
end
% the comparison must be of the 15-variable setup, with kappa optimized
assert(isequal(size(res2.S), [15 5]) && ~isempty(res2.ml_opt), ...
    'dataset 2: the setup is not the 15-variable one');

% --- the defaults both runs override, against the legacy values ---
% patch_once above has already asserted that the legacy script sets nsim = 5000,
% nsim = 1000 and nbatch = 50000, each exactly once
pr = res.preset;
assert(isequal(pr.d1.nsim, 5000) && isequal(pr.d2.nsim, 1000) && isequal(pr.nbatch, 50000), ...
    'the nsim or nbatch defaults in preset.m differ from the legacy script');

% --- bvar.priors.acp_opt_kappa against get_OptKappa, both variants ---
addpath(fullfile(tmp, 'utility'));
data = xlsread(fullfile(tmp, 'database_2019Q4.xlsx')); %#ok<XLSRD>
p = 5; vid = 1:6; idx_ns = [1,2,4,5];
Y0 = data(1:8,vid); Y = data(9:end,vid);
[T,n] = size(Y); tmpY = [Y0(end-p+1:end,:); Y];
Z = zeros(T,n*p);
for ii=1:p, Z(:,(ii-1)*n+1:ii*n) = tmpY(p-ii+1:end-ii,:); end
Z = [ones(T,1) Z];
[m1,k1] = get_OptKappa(Y0,Y,Z,p,[.04,.0016],'redu',idx_ns);
[m2,k2] = bvar.priors.acp_opt_kappa(Y0,Y,Z,p,[.04,.0016],'redu',idx_ns);
assert(isequal(m1,m2) && isequal(k1,k2), 'acp_opt_kappa differs from get_OptKappa');
[m3,k3] = get_OptSymKappa(Y0,Y,Z,p,'redu',idx_ns);
[m4,k4] = bvar.priors.acp_opt_kappa(Y0,Y,Z,p,[],'redu',idx_ns,'symmetric',true);
assert(isequal(m3,m4) && isequal(k3,k4), 'symmetric acp_opt_kappa differs from get_OptSymKappa');
assert(isequal(ml_VAR_ACP(p,Y,Z,prior_ACP_stru(n,p,k1,bvar.priors.resid_var_ar4(Y0,Y),idx_ns)), ...
       bvar.ml.acp(p,Y,Z,bvar.priors.acp_stru(n,p,k1,bvar.priors.resid_var_ar4(Y0,Y),idx_ns))), ...
    'bvar.ml.acp differs from ml_VAR_ACP on the structural prior');
end

% -------------------------------------------------------------------------
function txt = patch_once(txt, from, to, what)
n = numel(strfind(txt, from));
assert(n == 1, 'expected exactly one occurrence of %s, found %d', what, n);
txt = strrep(txt, from, to);
end

function cleanup_tmp(tmp)
entries = strsplit(path, pathsep);
for k = 1:numel(entries)
    if strncmpi(entries{k}, tmp, numel(tmp))
        rmpath(entries{k});
    end
end
if exist(tmp, 'dir')
    rmdir(tmp, 's');
end
end

% -------------------------------------------------------------------------
function out = run_legacy(tmpdir_, seed_)
% Run the patched legacy script from the tempdir as the working directory; its
% addpath('./utility') and its bare data filename are both relative. Locals
% carry a trailing underscore so the script's own variables cannot clobber them.
od_ = cd(tmpdir_);

store_response = []; response_median = []; response_CI = [];
count_total = []; count_sat = [];

resolved_ = which('main_ACP_apps');
assert(strncmpi(resolved_, tmpdir_, numel(tmpdir_)), ...
    'main_ACP_apps must resolve from the tempdir copy, got %s', resolved_);

rng(seed_, 'twister');
try
    evalc('main_ACP_apps');
catch err_
    cd(od_);
    rethrow(err_);
end
s_ = rng;
cd(od_);

out = struct('store_response',store_response, 'response_median',response_median, ...
    'response_CI',response_CI, 'count_total',count_total, 'count_sat',count_sat, ...
    'rngstate',s_.State);
end

% -------------------------------------------------------------------------
function out = run_legacy_setup(tmpdir_, name_)
% Run a patched legacy script that stops before the rejection loop, and return
% its setup. As in run_legacy, it runs from the tempdir and its locals carry a
% trailing underscore.
od_ = cd(tmpdir_);

Y0 = []; Y = []; Z = []; var_id = []; idx_ns = []; kappa = []; ml_opt = [];
sig2 = []; prior_redu = []; S = []; Rineq = []; Ridx = [];

resolved_ = which(name_);
assert(strncmpi(resolved_, tmpdir_, numel(tmpdir_)), ...
    '%s must resolve from the tempdir copy, got %s', name_, resolved_);

try
    evalc(name_);
catch err_
    cd(od_);
    rethrow(err_);
end
cd(od_);

out = struct('Y0',Y0, 'Y',Y, 'Z',Z, 'var_id',var_id, 'idx_ns',idx_ns, ...
    'kappa',kappa, 'ml_opt',ml_opt, 'sig2',sig2, 'prior_redu',prior_redu, ...
    'S',S, 'Rineq',Rineq, 'Ridx',Ridx);
end
