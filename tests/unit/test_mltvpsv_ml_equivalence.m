function test_mltvpsv_ml_equivalence
% seeded equivalence of replications/chan_eisenstat2018_jae_mltvpsv/run_ml.m with the
% cp_ml pipeline of main_tvpsv.m on the package's data: the legacy model script and
% then its ml_*.m routine on one rng stream, run from tempdir copies. Asserts isequal
% on the log marginal likelihood, its standard error and the terminal rng state, with
% bugcompat = true for the RS models, at the legacy defaults p = 2 and r = 2 for every
% model and at r = 3 for the RS models.
%
% The patches are those of test_mltvpsv_equivalence (the clock-seed lines) and the
% one-factor substitutions of one_factor_patch for the model scripts, SVRW.m and the
% integrated- and marginal-likelihood routines. The package reads its data with
% xlsread and a range argument, which needs Excel, so CI skips this test;
% test_mltvpsv_ml compares the routines on generated data on CI.
root = getappdata(0, 'bvar_repo_root');
leg = fullfile(root, 'replications', 'chan_eisenstat2018_jae_mltvpsv', 'legacy');
repdir = fullfile(root, 'replications', 'chan_eisenstat2018_jae_mltvpsv');
rel = 'chan_eisenstat2018_jae_mltvpsv/legacy/';

nsims = 80; burnin = 20; M = 20; seed = 20260923;

% --- tempdir: patched scripts and routines + verbatim helpers ---
tmp = tempname; mkdir(tmp);
ctmp = onCleanup(@() cleanup_tmp(tmp));
scripts = {'TVPSV', 'TVP', 'TVP_R1_SV', 'TVP_R2_SV', 'TVP_R3_SV', ...
    'VAR_SV', 'VAR', 'VAR_RS', 'VAR_RS_R1', 'VAR_RS_R2'};
routines = {'intlike_tvp.m', 'intlike_tvpsv.m', 'intlike_varsv.m', 'ml_tvpsv.m', 'ml_tvp.m', ...
    'ml_tvp_r1_sv.m', 'ml_tvp_r2_sv.m', 'ml_tvp_r3_sv.m', 'ml_varsv.m', 'ml_var.m', ...
    'ml_var_rs.m', 'ml_var_rs_r1.m', 'ml_var_rs_r2.m', 'SVRW.m'};
for f = [routines, {'intlike_var_rs.m', 'SURform.m', 'SURform2.m', 'constructX.m', ...
        'dirifit.m', 'dirirnd.m', 'ldiripdf.m'}]
    copyfile(fullfile(leg, f{1}), fullfile(tmp, f{1}));
end
for f = routines
    one_factor_patch(fullfile(tmp, f{1}), [rel f{1}]);
end
seedline = 'randn(''seed'',sum(clock*100)); rand(''seed'',sum(clock*1000));';
for k = 1:numel(scripts)
    f = [scripts{k} '.m'];
    txt = fileread(fullfile(leg, f));
    assert(numel(strfind(txt, seedline)) == 1, 'expected exactly one clock-seed line in %s', f);
    txt = strrep(txt, seedline, '% [clock-seed line removed by test_mltvpsv_ml_equivalence]');
    fid = fopen(fullfile(tmp, f), 'w');
    fwrite(fid, txt);
    fclose(fid);
    one_factor_patch(fullfile(tmp, f), [rel f]);
end
addpath(tmp);
addpath(repdir); cp2 = onCleanup(@() rmpath(repdir)); %#ok<NASGU>
assert(strncmpi(which('run_ml'), repdir, numel(repdir)), 'run_ml must resolve from the ml_tvpsv package');
for f = [scripts, {'ml_tvpsv', 'intlike_tvpsv', 'SVRW'}]
    w = which(f{1});
    assert(strncmpi(w, tmp, numel(tmp)), '%s must resolve from the tempdir copy, got %s', f{1}, w);
end

USdata = xlsread(fullfile(leg, 'USdata_2014Q4.xlsx'), 'B28:E269');   % main_tvpsv.m 35-36
data = USdata(:,[1 3 4]);

res = [];     % assigned inside evalc, which keeps the displays quiet
cases = [ (1:10)', 2*ones(10,1), 2*ones(10,1)
          8 1 3; 9 1 3; 10 1 3 ];
for kc = 1:size(cases, 1)
    imodel = cases(kc,1); p = cases(kc,2); r = cases(kc,3);
    tag = sprintf('model %d (%s, p=%d, r=%d)', imodel, scripts{imodel}, p, r);
    L = run_legacy(imodel, scripts{imodel}, data, p, r, nsims, burnin, M, seed);
    evalc('res = run_ml(imodel, p, r, nsims, burnin, seed, ''M'', M, ''bugcompat'', true);');
    sC = rng;
    assert(isequal(L.ml, res.lml) && isequal(L.ml_std, res.lmlstd), ...
        '%s: %.15g (%.15g) against %.15g (%.15g)', tag, res.lml, res.lmlstd, L.ml, L.ml_std);
    assert(isequal(L.rngstate, sC.State), '%s: rng call sequence differs', tag);
end
end

% -------------------------------------------------------------------------
function cleanup_tmp(tmp)
if any(strcmpi(strsplit(path, pathsep), tmp))
    rmpath(tmp);
end
if exist(tmp, 'dir')
    rmdir(tmp, 's');
end
end

% -------------------------------------------------------------------------
function out = run_legacy(model, script, data, p, r, nsims, burnin, M, seed) %#ok<INUSL>
% replicate main_tvpsv.m lines 35-80 in this workspace: the data, the (patched) model
% script from the tempdir and its ml_*.m routine, on the stream seeded here
Y0 = data(1:4,:);
shortY = data(5:end,:);
[T,n] = size(shortY);       %#ok<ASGLU>
Y = reshape(shortY',T*n,1);

prior = []; bigX = []; Xtilde = []; W = []; Z = [];
store_theta = []; store_Sigtheta = []; store_Sigh = []; store_theta0 = []; store_h0 = [];
store_Sig = []; store_beta = []; store_gam = []; store_Siggam = []; store_gam0 = [];
store_Sigbeta = []; store_beta0 = []; store_Sigmu = []; store_mu0 = []; store_P = [];
ml = []; ml_std = [];

rng(seed, 'twister');
evalc(script);
switch model
    case 1
        evalc(['[ml, ml_std] = ml_tvpsv(Y,store_Sigtheta,store_Sigh,store_h0,' ...
            'store_theta0,prior,bigX,M);']);
    case 2
        evalc('[ml, ml_std] = ml_tvp(Y,store_Sig,store_Sigtheta,store_theta0,prior,bigX,M);');
    case 3
        evalc(['[ml, ml_std] = ml_tvp_r1_sv(Y,store_beta,store_Siggam,store_Sigh,' ...
            'store_h0,store_gam0,prior,Xtilde,W,M);']);
    case 4
        evalc(['[ml, ml_std] = ml_tvp_r2_sv(Y,store_gam,store_Sigbeta,store_Sigh,' ...
            'store_h0,store_beta0,prior,Xtilde,W,M);']);
    case 5
        evalc(['[ml,ml_std] = ml_tvp_r3_sv(Y,store_beta,store_gam,store_Sigmu,store_Sigh,' ...
            'store_h0,store_mu0,prior,Z,M);']);
    case 6
        evalc('[ml, ml_std] = ml_varsv(Y,store_theta,store_Sigh,store_h0,prior,bigX,M);');
    case 7
        evalc('[ml, ml_std] = ml_var(Y,store_theta,store_Sig,prior,bigX,M);');
    case 8
        evalc('[ml,ml_std] = ml_var_rs(Y,store_theta,store_Sig,store_P,prior,bigX,M);');
    case 9
        evalc('[ml,ml_std] = ml_var_rs_r1(Y,store_theta,store_Sig,store_P,prior,bigX,M);');
    case 10
        evalc('[ml,ml_std] = ml_var_rs_r2(Y,store_theta,store_Sig,store_P,prior,bigX,M);');
end
s = rng;
out = struct('ml', ml, 'ml_std', ml_std, 'rngstate', s.State);
end
