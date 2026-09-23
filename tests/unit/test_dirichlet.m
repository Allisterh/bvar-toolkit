function test_dirichlet
% bvar.util.dirirnd, bvar.ml.ldiripdf and bvar.ml.dirifit against the ml_tvpsv
% package's dirirnd.m, ldiripdf.m and dirifit.m, run from tempdir copies: identical
% draws under a seed, identical densities and fits. Also checks the density against
% its closed form, and that the fit recovers the parameters of a large sample.
root = getappdata(0, 'bvar_repo_root');
leg = fullfile(root, 'replications', 'chan_eisenstat2018_jae_mltvpsv', 'legacy');
tmp = tempname; mkdir(tmp);
ctmp = onCleanup(@() cleanup_tmp(tmp));
for f = {'dirirnd.m', 'ldiripdf.m', 'dirifit.m'}
    copyfile(fullfile(leg, f{1}), fullfile(tmp, f{1}));
end
addpath(tmp);
assert(strncmpi(which('dirifit'), tmp, numel(tmp)), 'dirifit must resolve from the tempdir copy');

    % draws: one and many, r = 2, 3, 6
for alp = {[2; 2], [2; 5; 1.5], (1:6)'/2}
    for N = [1 7 1000]
        rng(31, 'twister'); if N == 1, L = dirirnd(alp{1}); else, L = dirirnd(alp{1}, N); end
        rng(31, 'twister'); if N == 1, C = bvar.util.dirirnd(alp{1}); else, C = bvar.util.dirirnd(alp{1}, N); end
        assert(isequal(L, C), 'dirirnd differs (r = %d, N = %d)', numel(alp{1}), N);
        assert(all(abs(sum(C, 2) - 1) < 1e-12) && all(C(:) > 0), 'dirirnd: rows off the simplex');
    end
end

    % densities: alpha as a row and as a column
rng(32, 'twister');
y = bvar.util.dirirnd([3; 1; 2], 50);
for alpha = {[3 1 2], [3; 1; 2], [.7 1.4 2.5]}
    assert(isequal(ldiripdf(y, alpha{1}), bvar.ml.ldiripdf(y, alpha{1})), 'ldiripdf differs');
end
a = [.7; 1.4; 2.5];
ref = gammaln(sum(a)) - sum(gammaln(a)) + log(y)*(a - 1);
assert(max(abs(bvar.ml.ldiripdf(y, a) - ref)) < 1e-12, 'ldiripdf: closed form');
try
    bvar.ml.ldiripdf(y, [1 2]);
    error('test:noError', 'ldiripdf accepted mismatched dimensions');
catch err
    assert(strcmp(err.message, 'dimensions do not match '), 'ldiripdf: wrong error');
end

    % fits: identical to the legacy, and near the truth in a large sample
for alp = {[2; 2], [2; 5; 1.5], [8; 3; 4; 1]}
    rng(33, 'twister');
    X = bvar.util.dirirnd(alp{1}, 20000);
    [aL, fL] = dirifit(X);
    [aC, fC] = bvar.ml.dirifit(X);
    assert(isequal(aL, aC) && isequal(fL, fC), 'dirifit differs');
    assert(fC == 1 && max(abs(aC - alp{1})./alp{1}) < .05, 'dirifit: far from the truth');
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
