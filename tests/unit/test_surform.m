function test_surform
% structure + exact equivalence with the legacy HYB and ml_tvpsv copies
rng(1, 'twister');
X = randn(7, 3);
Z = bvar.util.surform(X);
assert(isequal(size(Z), [7, 21]), 'surform: wrong size');
assert(isequal(full(Z(2, 4:6)), X(2, :)), 'surform: wrong block placement');
assert(nnz(Z) == numel(X), 'surform: wrong sparsity');

root = getappdata(0, 'bvar_repo_root');
leg = fullfile(root, 'replications', 'chan2023_jbes_hybtvp', 'legacy', 'utility');
addpath(leg); c = onCleanup(@() rmpath(leg));
assert(isequal(SURform(X), Z), 'surform: differs from legacy SURform');
clear c                                          % the HYB copy leaves the path first

leg = fullfile(root, 'replications', 'chan_eisenstat2018_jae_mltvpsv', 'legacy');
addpath(leg); c = onCleanup(@() rmpath(leg));
assert(isequal(SURform(X), Z), 'surform: differs from the ml_tvpsv SURform');
end
