function test_mltvpsv_options
% replications/chan_eisenstat2018_jae_mltvpsv/run_all on generated data, so that it
% runs on CI, where test_mltvpsv_equivalence is skipped: every model runs, the stored
% draws have the documented sizes and repeat under a seed, the paper's labels, the
% legacy spellings and the numeric codes select the same model, and bad inputs raise
% the documented errors.
root = getappdata(0, 'bvar_repo_root');
repdir = fullfile(root, 'replications', 'chan_eisenstat2018_jae_mltvpsv');
addpath(repdir); cp = onCleanup(@() rmpath(repdir)); %#ok<NASGU>
resolved = which('run_all');
assert(strncmpi(resolved, repdir, numel(repdir)), ...
    'run_all must resolve from the ml_tvpsv package, got %s', resolved);

% generated data: a stationary VAR(1) in three variables, 4 initial rows + 60
rng(7, 'twister');
n = 3; T0 = 64;
D = zeros(T0, n);
for t = 2:T0
    D(t,:) = [.5 .2 .1] + D(t-1,:)*diag([.6 .5 .4]) + .5*randn(1,n);
end
T = T0 - 4;

names = {'TVP-SV','TVP','TVP-R1-SV','TVP-R2-SV','TVP-R3-SV','CVAR-SV','CVAR', ...
    'RS-VAR','RS-VAR-R1','RS-VAR-R2'};
nsims = 4; burnin = 2; p = 2; r = 2;
m = n*(n-1)/2;
for imodel = 1:10
    res = run_all(names{imodel}, p, r, nsims, burnin, 11, 'data', D);
    res2 = run_all(imodel, p, r, nsims, burnin, 11, 'data', D);
    assert(strcmp(res.model, names{imodel}) && res.model_num == imodel, '%s: label', names{imodel});
    assert(isequal(rmfield(res, {'prior','preset'}), rmfield(res2, {'prior','preset'})), ...
        '%s: the label and the code give different runs', names{imodel});
    assert(res.T == T && res.n == n && numel(res.Y) == T*n, '%s: data dimensions', names{imodel});
    switch imodel
        case {1, 2}
            k = n^2*p + n + m;
            assert(isequal(size(res.store_theta), [nsims T*k]), '%s: store_theta', names{imodel});
        case {3, 4}
            assert(size(res.store_gam, 1) == nsims && size(res.store_beta, 1) == nsims, ...
                '%s: stored draws', names{imodel});
        case 5
            assert(isequal(size(res.store_mu), [nsims T*n]), '%s: store_mu', names{imodel});
        case {6, 7}
            assert(isequal(size(res.store_theta), [nsims n^2*p+n+m]), '%s: store_theta', names{imodel});
        otherwise
            assert(isequal(size(res.store_P), [nsims r r]) && isequal(size(res.store_S), [T r]), ...
                '%s: regime draws', names{imodel});
    end
end

% the legacy spellings of models 6, 7, 9 and 10
aliases = {'VAR-SV', 6; 'VAR', 7; 'RS-VAR-1', 9; 'RS-VAR-2', 10};
for ka = 1:size(aliases, 1)
    res = run_all(aliases{ka,1}, 1, 2, 2, 1, 3, 'data', D);
    assert(res.model_num == aliases{ka,2}, 'alias %s', aliases{ka,1});
end

% bad inputs
bad = { {11}, 'run_all:model'; {'VAR-XYZ'}, 'run_all:model'
        {6, 0}, 'run_all:badOption'; {6, 5}, 'run_all:badOption'
        {8, 2, 1}, 'run_all:badOption'; {6, 2, 2, 0}, 'run_all:badOption'
        {6, 2, 2, 10, -1}, 'run_all:badOption'
        {6, 2, 2, 10, 5, 1, 'data'}, 'run_all:badOption'
        {6, 2, 2, 10, 5, 1, 'nsim', 3}, 'run_all:badOption'
        {6, 2, 2, 10, 5, 1, 'data', D(1:5,:)}, 'run_all:badData'
        {6, 2, 2, 10, 5, 1, 'data', [D(:,1:2); NaN NaN]}, 'run_all:badData' };
for kb = 1:size(bad, 1)
    try
        run_all(bad{kb,1}{:});
        error('test:noError', 'case %d raised no error', kb);
    catch err
        assert(strcmp(err.identifier, bad{kb,2}), 'case %d raised %s', kb, err.identifier);
    end
end
end
