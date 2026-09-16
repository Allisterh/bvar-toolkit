function test_surform2
% structure + exact equivalence with the legacy ml_varsv and ml_tvpsv copies
rng(2, 'twister');
n = 4; X = randn(6, 3);
Z = bvar.util.surform2(X, n);
assert(isequal(size(Z), [6*n, 3*n]), 'surform2: wrong size');
assert(isequal(full(Z(1:n, :)), kron(eye(n), X(1, :))), 'surform2: wrong first block');

root = getappdata(0, 'bvar_repo_root');
leg = fullfile(root, 'replications', 'chan2023_joe_mlvarsv', 'legacy', 'utility');
addpath(leg); c = onCleanup(@() rmpath(leg));
assert(isequal(SURform2(X, n), Z), 'surform2: differs from legacy SURform2');
clear c                                          % the ml_varsv copy leaves the path first

leg = fullfile(root, 'replications', 'chan_eisenstat2018_jae_mltvpsv', 'legacy');
addpath(leg); c = onCleanup(@() rmpath(leg));
assert(isequal(SURform2(X, n), Z), 'surform2: differs from the ml_tvpsv SURform2');
end
