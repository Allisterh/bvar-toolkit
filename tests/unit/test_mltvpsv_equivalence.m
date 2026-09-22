function test_mltvpsv_equivalence
% seeded draw-for-draw equivalence of the functionized ml_tvpsv estimation pipeline
% (replications/chan_eisenstat2018_jae_mltvpsv/run_all.m + bvar.sv.ksc_rw_h0,
% bvar.util.surform, surform2 and build_lags) with the ten legacy workspace scripts,
% run from tempdir copies at small nsims on the package's data. Asserts isequal on all
% stored draws, the script-tail summaries, the design matrices, the log prior at the
% last draw, and the terminal rng state.
%
% Two patches to the legacy copies. Each script carries
%   randn('seed',sum(clock*100)); rand('seed',sum(clock*1000));
% active (asserted: exactly one occurrence each, not commented) - it re-seeds from the
% wall clock and switches MATLAB to the v4/v5 generators, so it is removed. Every rng
% draw sits after that point, so rng(seed,'twister') before dispatch aligns the whole
% run. The scripts and SVRW.m get the one-factor substitutions of one_factor_patch.
%
% The package reads its data with xlsread and a range argument, which needs Excel, so
% CI skips this test (BVAR_SKIP_TESTS); test_mltvpsv_options covers run_all on CI.
root = getappdata(0, 'bvar_repo_root');
leg = fullfile(root, 'replications', 'chan_eisenstat2018_jae_mltvpsv', 'legacy');
repdir = fullfile(root, 'replications', 'chan_eisenstat2018_jae_mltvpsv');
rel = 'chan_eisenstat2018_jae_mltvpsv/legacy/';

nsims = 30; burnin = 10; seed = 20260922;

% --- tempdir: patched model scripts + verbatim helpers ---
tmp = tempname; mkdir(tmp);
ctmp = onCleanup(@() cleanup_tmp(tmp));   % rmpath BEFORE rmdir, warning-free
for f = {'SURform.m', 'SURform2.m', 'constructX.m', 'dirirnd.m', 'ldiripdf.m', 'SVRW.m'}
    copyfile(fullfile(leg, f{1}), fullfile(tmp, f{1}));
end
one_factor_patch(fullfile(tmp, 'SVRW.m'), [rel 'SVRW.m']);

scripts = {'TVPSV', 'TVP', 'TVP_R1_SV', 'TVP_R2_SV', 'TVP_R3_SV', ...
    'VAR_SV', 'VAR', 'VAR_RS', 'VAR_RS_R1', 'VAR_RS_R2'};
seedline = 'randn(''seed'',sum(clock*100)); rand(''seed'',sum(clock*1000));';
for k = 1:numel(scripts)
    f = [scripts{k} '.m'];
    txt = fileread(fullfile(leg, f));
    assert(numel(strfind(txt, seedline)) == 1, 'expected exactly one clock-seed line in %s', f);
    assert(isempty(strfind(txt, ['%' seedline])), 'the %s clock-seed line is expected to be ACTIVE', f);
    txt = strrep(txt, seedline, '% [clock-seed line removed by test_mltvpsv_equivalence]');
    fid = fopen(fullfile(tmp, f), 'w');
    fwrite(fid, txt);
    fclose(fid);
    one_factor_patch(fullfile(tmp, f), [rel f]);
end

addpath(tmp);   % removed by cleanup_tmp via ctmp, before the folder is deleted
addpath(repdir); cp2 = onCleanup(@() rmpath(repdir)); %#ok<NASGU>

% several packages define run_all - pin the resolution
resolved = which('run_all');
assert(strncmpi(resolved, repdir, numel(repdir)), ...
    'run_all must resolve from the ml_tvpsv package, got %s', resolved);
for f = [scripts, {'SVRW', 'SURform', 'SURform2', 'constructX', 'dirirnd', 'ldiripdf'}]
    w = which(f{1});
    assert(strncmpi(w, tmp, numel(tmp)), '%s must resolve from the tempdir copy, got %s', f{1}, w);
end

% the data, as main_tvpsv.m lines 35-41 read it
USdata = xlsread(fullfile(leg, 'USdata_2014Q4.xlsx'), 'B28:E269');
data = USdata(:,[1 3 4]);

% fields compared per model, and the arguments of its log prior (the order in which
% the ml_*.m routines pass them)
f = cell(10,1); a = cell(10,1);
f{1}  = {'store_theta','store_h','store_Sigtheta','store_Sigh','store_theta0','store_h0','bigX'};
a{1}  = {'store_Sigtheta','store_Sigh','store_theta0','store_h0'};
f{2}  = {'store_theta','store_Sigtheta','store_Sig','store_theta0','bigX'};
a{2}  = {'store_Sig','store_Sigtheta','store_theta0'};
f{3}  = {'store_beta','store_gam','store_h','store_Siggam','store_Sigh','store_gam0','store_h0', ...
         'betahat','betaCI','gamhat','gamCI','Xtilde','W'};
a{3}  = {'store_beta','store_Siggam','store_Sigh','store_gam0','store_h0'};
f{4}  = {'store_beta','store_gam','store_h','store_Sigbeta','store_Sigh','store_beta0','store_h0', ...
         'betahat','betaCI','gamhat','gamCI','Xtilde','W'};
a{4}  = {'store_gam','store_Sigbeta','store_Sigh','store_beta0','store_h0'};
f{5}  = {'store_mu','store_beta','store_gam','store_h','store_Sigmu','store_Sigh','store_mu0', ...
         'store_h0','muhat','muCI','beta_hat','gam_hat','gamCI','Ztilde','W','Z'};
a{5}  = {'store_beta','store_gam','store_Sigmu','store_Sigh','store_mu0','store_h0'};
f{6}  = {'store_theta','store_h','store_Sigh','store_h0','thetahat','thetaCI','hhat','hCI','bigX'};
a{6}  = {'store_theta','store_Sigh','store_h0'};
f{7}  = {'store_theta','store_Sig','theta_hat','thetaCI','bigX'};
a{7}  = {'store_theta','store_Sig'};
f{8}  = {'store_theta','store_Sig','store_P','store_S','theta_hat','thetaCI','S_hat','bigX'};
a{8}  = {'store_theta','store_Sig','store_P'};
f{9}  = f{8}; a{9} = a{8};
f{10} = f{8}; a{10} = a{8};

% model, p, r. Every model at the legacy default p = 2 (r = 2 for the RS models), the
% lag bounds p = 1 and p = 4 on a model of each design, and the RS models at r = 3 and
% at r = 12, where regimes empty out and the prior-draw branches run (at r = 6
% RS-VAR-R1 and RS-VAR-R2 never visit an empty regime on this data).
cases = [ (1:10)', 2*ones(10,1), 2*ones(10,1)
          1 1 2; 1 4 2; 3 4 2; 4 1 2; 5 4 2; 6 4 2; 7 1 2; 8 4 2
          8 2 3; 9 2 3; 10 2 3
          8 1 12; 9 1 12; 10 1 12 ];

nempty = zeros(10,1);
for kc = 1:size(cases, 1)
    imodel = cases(kc,1); p = cases(kc,2); r = cases(kc,3);
    tag = sprintf('model %d (%s, p=%d, r=%d)', imodel, scripts{imodel}, p, r);

    L = run_legacy(scripts{imodel}, data, p, r, nsims, burnin, seed);
    res = run_all(imodel, p, r, nsims, burnin, seed);   % re-seeds itself
    sC = rng;

    for kf = 1:numel(f{imodel})
        fld = f{imodel}{kf};
        assert(isequal(L.(fld), res.(fld)), '%s: %s differs', tag, fld);
    end
    assert(isequal(L.rngstate, sC.State), '%s: rng call sequence differs', tag);

        % the log prior at the last stored draw, through both handles
    args = cell(1, numel(a{imodel}));
    for ka = 1:numel(args)
        v = L.(a{imodel}{ka});
        if ndims(v) == 3
            args{ka} = squeeze(v(end,:,:));   % the transition matrix P
        else
            args{ka} = v(end,:)';
        end
    end
    lp_leg = L.prior(args{:});
    lp_new = res.prior(args{:});
    assert(isfinite(lp_leg) && isequal(lp_leg, lp_new), '%s: log prior differs', tag);

    if imodel >= 8
        nempty(imodel) = nempty(imodel) + res.count_empty;
    end
end
% the empty-regime branches must actually run, or they are untested
assert(all(nempty(8:10) > 0), 'an RS model never visited an empty regime: %s', mat2str(nempty(8:10)'));

% the 'data' option reproduces the package's own file
res = run_all(6, 2, 2, 5, 2, seed);
res2 = run_all(6, 2, 2, 5, 2, seed, 'data', data);
assert(isequal(res.store_theta, res2.store_theta) && isequal(res.Y, res2.Y), ...
    'the data option does not reproduce the package data');
end

% -------------------------------------------------------------------------
function cleanup_tmp(tmp)
% deterministic teardown order: path entry first, then the folder itself
if any(strcmpi(strsplit(path, pathsep), tmp))
    rmpath(tmp);
end
if exist(tmp, 'dir')
    rmdir(tmp, 's');
end
end

% -------------------------------------------------------------------------
function out = run_legacy(script, data, p, r, nsims, burnin, seed) %#ok<INUSL>
% replicate main_tvpsv.m lines 35-41 in this workspace and dispatch the (patched)
% legacy script from the tempdir. p, r, nsims and burnin are read by the dispatched
% script from here, hence the INUSL suppression above.
Y0 = data(1:4,:);  % GDP deflator growth, real GDP growth, Fed funds rate
shortY = data(5:end,:);
[T,n] = size(shortY);       %#ok<ASGLU>
Y = reshape(shortY',T*n,1); %#ok<NASGU>

% pre-declare the variables the dispatched script assigns and we read back, so the
% parser binds them as variables in this workspace
prior = []; bigX = []; Xtilde = []; Ztilde = []; W = []; Z = [];
store_theta = []; store_h = []; store_Sigtheta = []; store_Sigh = []; store_theta0 = [];
store_h0 = []; store_Sig = []; store_beta = []; store_gam = []; store_Siggam = [];
store_gam0 = []; store_Sigbeta = []; store_beta0 = []; store_mu = []; store_Sigmu = [];
store_mu0 = []; store_P = []; store_S = [];
betahat = []; betaCI = []; gamhat = []; gamCI = []; muhat = []; muCI = [];
beta_hat = []; gam_hat = []; thetahat = []; thetaCI = []; hhat = []; hCI = [];
theta_hat = []; S_hat = [];

rng(seed, 'twister');
switch script
    case 'TVPSV',     TVPSV;
    case 'TVP',       TVP;
    case 'TVP_R1_SV', TVP_R1_SV;
    case 'TVP_R2_SV', TVP_R2_SV;
    case 'TVP_R3_SV', TVP_R3_SV;
    case 'VAR_SV',    VAR_SV;
    case 'VAR',       VAR;
    case 'VAR_RS',    VAR_RS;
    case 'VAR_RS_R1', VAR_RS_R1;
    case 'VAR_RS_R2', VAR_RS_R2;
end
s = rng;

out = struct('prior',prior, 'bigX',bigX, 'Xtilde',Xtilde, 'Ztilde',Ztilde, 'W',W, 'Z',Z, ...
    'store_theta',store_theta, 'store_h',store_h, 'store_Sigtheta',store_Sigtheta, ...
    'store_Sigh',store_Sigh, 'store_theta0',store_theta0, 'store_h0',store_h0, ...
    'store_Sig',store_Sig, 'store_beta',store_beta, 'store_gam',store_gam, ...
    'store_Siggam',store_Siggam, 'store_gam0',store_gam0, 'store_Sigbeta',store_Sigbeta, ...
    'store_beta0',store_beta0, 'store_mu',store_mu, 'store_Sigmu',store_Sigmu, ...
    'store_mu0',store_mu0, 'store_P',store_P, 'store_S',store_S, ...
    'betahat',betahat, 'betaCI',betaCI, 'gamhat',gamhat, 'gamCI',gamCI, ...
    'muhat',muhat, 'muCI',muCI, 'beta_hat',beta_hat, 'gam_hat',gam_hat, ...
    'thetahat',thetahat, 'thetaCI',thetaCI, 'hhat',hhat, 'hCI',hCI, ...
    'theta_hat',theta_hat, 'S_hat',S_hat, 'rngstate',s.State);
end
