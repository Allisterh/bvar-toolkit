function test_mltvpsv_ml
% the ten bvar.ml.mltvpsv_* routines and bvar.ml.intlike_tvp, intlike_tvpsv,
% intlike_cvarsv and intlike_rsvar against the ml_tvpsv package's ml_*.m and
% intlike_*.m, run from tempdir copies with the one-factor substitutions, on posterior
% draws that run_all produces from generated data: identical log marginal likelihoods,
% standard errors and terminal rng states under a seed, with bugcompat = true for the
% RS models. With bugcompat = false the RS estimates change by exactly log(3/r)
% (nothing at r = 3). The integrated likelihoods are also compared directly, with a given number
% of draws and on the path where the draws increase adaptively.
root = getappdata(0, 'bvar_repo_root');
leg = fullfile(root, 'replications', 'chan_eisenstat2018_jae_mltvpsv', 'legacy');
repdir = fullfile(root, 'replications', 'chan_eisenstat2018_jae_mltvpsv');
rel = 'chan_eisenstat2018_jae_mltvpsv/legacy/';

tmp = tempname; mkdir(tmp);
ctmp = onCleanup(@() cleanup_tmp(tmp));
patched = {'intlike_tvp.m', 'intlike_tvpsv.m', 'intlike_varsv.m', 'ml_tvpsv.m', 'ml_tvp.m', ...
    'ml_tvp_r1_sv.m', 'ml_tvp_r2_sv.m', 'ml_tvp_r3_sv.m', 'ml_varsv.m', 'ml_var.m', ...
    'ml_var_rs.m', 'ml_var_rs_r1.m', 'ml_var_rs_r2.m'};
for f = [patched, {'intlike_var_rs.m', 'dirifit.m', 'dirirnd.m', 'ldiripdf.m'}]
    copyfile(fullfile(leg, f{1}), fullfile(tmp, f{1}));
end
for f = patched
    one_factor_patch(fullfile(tmp, f{1}), [rel f{1}]);
end
addpath(tmp);
addpath(repdir); cp2 = onCleanup(@() rmpath(repdir)); %#ok<NASGU>
assert(strncmpi(which('run_all'), repdir, numel(repdir)), 'run_all must resolve from the ml_tvpsv package');
assert(strncmpi(which('ml_var_rs'), tmp, numel(tmp)), 'ml_var_rs must resolve from the tempdir copy');

% generated data: a stationary VAR(1) in three variables, 4 initial rows + 80
rng(9, 'twister');
n = 3; T0 = 84;
D = zeros(T0, n);
for t = 2:T0
    D(t,:) = [.5 .2 .1] + D(t-1,:)*diag([.6 .5 .4]) + .5*randn(1,n);
end

M = 40; seed = 424242;
mlL = []; sdL = [];     % assigned inside evalc, which keeps the legacy displays quiet
legacy = {'ml_tvpsv', 'ml_tvp', 'ml_tvp_r1_sv', 'ml_tvp_r2_sv', 'ml_tvp_r3_sv', 'ml_varsv', ...
    'ml_var', 'ml_var_rs', 'ml_var_rs_r1', 'ml_var_rs_r2'};
core = {'mltvpsv_tvpsv', 'mltvpsv_tvp', 'mltvpsv_tvp_r1_sv', 'mltvpsv_tvp_r2_sv', ...
    'mltvpsv_tvp_r3_sv', 'mltvpsv_cvarsv', 'mltvpsv_cvar', 'mltvpsv_rs', 'mltvpsv_rs_r1', ...
    'mltvpsv_rs_r2'};
for imodel = 1:10
    for r = unique([2, 3*(imodel >= 8) + 2*(imodel < 8)])
        e = run_all(imodel, 1, r, 60, 20, 5, 'data', D);
        args = ml_args(imodel, e, M);
        rng(seed, 'twister'); evalc('[mlL, sdL] = feval(legacy{imodel}, args{:});'); sL = rng;
        if imodel >= 8
            rng(seed, 'twister'); [mlC, sdC] = feval(['bvar.ml.' core{imodel}], args{:}, true); sC = rng;
            rng(seed, 'twister'); [mlF, sdF] = feval(['bvar.ml.' core{imodel}], args{:}); sF = rng;
            assert(isequal(sC.State, sF.State), '%s: bugcompat changes the rng use', core{imodel});
            if r == 3
                assert(isequal(mlF, mlC) && isequal(sdF, sdC), '%s: r = 3 must not depend on bugcompat', core{imodel});
            else
                assert(abs((mlF - mlC) - log(3/r)) < 1e-9 && abs(sdF - sdC) < 1e-9, ...
                    '%s: the corrected estimate is not the published one + log(3/r)', core{imodel});
            end
        else
            rng(seed, 'twister'); [mlC, sdC] = feval(['bvar.ml.' core{imodel}], args{:}); sC = rng;
        end
        assert(isequal(mlL, mlC) && isequal(sdL, sdC), '%s (r = %d): %.15g (%.15g) against %.15g (%.15g)', ...
            core{imodel}, r, mlC, sdC, mlL, sdL);
        assert(isequal(sL.State, sC.State), '%s: rng call sequence differs', core{imodel});
        assert(isfinite(mlC) && sdC >= 0, '%s: estimate not finite', core{imodel});
    end
end

    % the integrated likelihoods directly, at the posterior mean and away from it; at the
    % second point (Sigtheta/100, Sigh*1000) intlike_tvpsv increases its draws about
    % thirteen-fold, and its Newton step warns of a nearly singular Hessian
e1 = run_all(1, 1, 2, 60, 20, 5, 'data', D);
e6 = run_all(6, 1, 2, 60, 20, 5, 'data', D);
expanded = false;
pts = [1 1; .01 1000; 4 16];
ws = warning('off', 'MATLAB:nearlySingularMatrix');
cw = onCleanup(@() warning(ws)); %#ok<NASGU>
for kp = 1:size(pts, 1)
    a = pts(kp,1); b = pts(kp,2); scale = kp;
    Sigtheta = a*mean(e1.store_Sigtheta)'; Sigh = b*mean(e1.store_Sigh)';
    h0 = mean(e1.store_h0)'; theta0 = mean(e1.store_theta0)';
    rng(scale, 'twister'); [lL, sL] = intlike_tvpsv(e1.Y, Sigtheta, Sigh, e1.bigX, h0, theta0); stL = rng;
    rng(scale, 'twister'); [lC, sC] = bvar.ml.intlike_tvpsv(e1.Y, Sigtheta, Sigh, e1.bigX, h0, theta0); stC = rng;
    assert(isequal(lL, lC) && isequal(sL, sC) && isequal(stL.State, stC.State), 'intlike_tvpsv differs (scale %d)', scale);
    expanded = expanded || numel(sC) > 10;
    rng(scale, 'twister'); lL = intlike_tvpsv(e1.Y, Sigtheta, Sigh, e1.bigX, h0, theta0, 7);
    rng(scale, 'twister'); lC = bvar.ml.intlike_tvpsv(e1.Y, Sigtheta, Sigh, e1.bigX, h0, theta0, 7);
    assert(isequal(lL, lC), 'intlike_tvpsv with R = 7 differs (scale %d)', scale);

    theta = mean(e6.store_theta)'; Sigh6 = b*mean(e6.store_Sigh)'; h06 = mean(e6.store_h0)';
    rng(scale, 'twister'); lL = intlike_varsv(e6.Y, theta, Sigh6, e6.bigX, h06); stL = rng;
    rng(scale, 'twister'); lC = bvar.ml.intlike_cvarsv(e6.Y, theta, Sigh6, e6.bigX, h06); stC = rng;
    assert(isequal(lL, lC) && isequal(stL.State, stC.State), 'intlike_cvarsv differs (scale %d)', scale);
    rng(scale, 'twister'); lL = intlike_varsv(e6.Y, theta, Sigh6, e6.bigX, h06, 7);
    rng(scale, 'twister'); lC = bvar.ml.intlike_cvarsv(e6.Y, theta, Sigh6, e6.bigX, h06, 7);
    assert(isequal(lL, lC), 'intlike_cvarsv with R = 7 differs (scale %d)', scale);
end
assert(expanded, 'intlike_tvpsv never increased its draws: the adaptive branch is untested');

e2 = run_all(2, 1, 2, 60, 20, 5, 'data', D);
for scale = [1 4]
    lL = intlike_tvp(e2.Y, scale*mean(e2.store_Sig)', mean(e2.store_Sigtheta)', e2.bigX, mean(e2.store_theta0)');
    lC = bvar.ml.intlike_tvp(e2.Y, scale*mean(e2.store_Sig)', mean(e2.store_Sigtheta)', e2.bigX, mean(e2.store_theta0)');
    assert(isequal(lL, lC), 'intlike_tvp differs (scale %d)', scale);
end
e8 = run_all(8, 1, 3, 60, 20, 5, 'data', D);
k8 = size(e8.store_theta, 2)/3;
P = squeeze(mean(e8.store_P, 1));
th = reshape(mean(e8.store_theta)', k8, 3); sg = reshape(mean(e8.store_Sig)', n, 3);
lL = intlike_var_rs(e8.shortY, e8.bigX, th, sg, P);
lC = bvar.ml.intlike_rsvar(e8.shortY, e8.bigX, th, sg, P);
assert(isequal(lL, lC), 'intlike_rsvar differs at r = 3');
end

% -------------------------------------------------------------------------
function args = ml_args(imodel, e, M)
% the arguments main_tvpsv.m passes to each ml_*.m (lines 48-80)
switch imodel
    case 1, args = {e.Y, e.store_Sigtheta, e.store_Sigh, e.store_h0, e.store_theta0, e.prior, e.bigX, M};
    case 2, args = {e.Y, e.store_Sig, e.store_Sigtheta, e.store_theta0, e.prior, e.bigX, M};
    case 3, args = {e.Y, e.store_beta, e.store_Siggam, e.store_Sigh, e.store_h0, e.store_gam0, e.prior, e.Xtilde, e.W, M};
    case 4, args = {e.Y, e.store_gam, e.store_Sigbeta, e.store_Sigh, e.store_h0, e.store_beta0, e.prior, e.Xtilde, e.W, M};
    case 5, args = {e.Y, e.store_beta, e.store_gam, e.store_Sigmu, e.store_Sigh, e.store_h0, e.store_mu0, e.prior, e.Z, M};
    case 6, args = {e.Y, e.store_theta, e.store_Sigh, e.store_h0, e.prior, e.bigX, M};
    case 7, args = {e.Y, e.store_theta, e.store_Sig, e.prior, e.bigX, M};
    otherwise, args = {e.Y, e.store_theta, e.store_Sig, e.store_P, e.prior, e.bigX, M};
end
end

function cleanup_tmp(tmp)
if any(strcmpi(strsplit(path, pathsep), tmp))
    rmpath(tmp);
end
if exist(tmp, 'dir')
    rmdir(tmp, 's');
end
end
