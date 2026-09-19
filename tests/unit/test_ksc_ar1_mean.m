function test_ksc_ar1_mean
% bvar.sv.ksc_ar1_mean must reproduce, draw-for-draw under one seed, BOTH legacy
% sample_SV.m copies it canonicalizes (ml_varsv and OISV) - path AND indicators -
% each run from a tempdir copy with the one-factor substitution of
% one_factor_patch.
root = getappdata(0, 'bvar_repo_root');

rng(43, 'twister');                          % fixed test data
T = 173;
e = randn(T,1).*exp(0.3*randn(T,1));
ystar = log(e.^2 + 1e-4);
h_in = -1 + 0.2*randn(T,1);
mu = -1.2; rho = 0.95; sig2 = 0.15;

rng(9, 'twister');
[h_core, S_core] = bvar.sv.ksc_ar1_mean(ystar, h_in, mu, rho, sig2);

copies = { ...
    'chan2023_joe_mlvarsv/legacy/utility/sample_SV.m'; ...
    'chan_koop_yu2024_jbes_oisv/legacy/utility/sample_SV.m'};

for ii = 1:numel(copies)
    tmp = tempname; mkdir(tmp);
    copyfile(fullfile(root, 'replications', copies{ii}), tmp);
    one_factor_patch(fullfile(tmp, 'sample_SV.m'), copies{ii});
    addpath(tmp); c = onCleanup(@() cleanup_tmp(tmp));  % prepended -> patched copy shadows
    rng(9, 'twister');
    [h_leg, S_leg] = sample_SV(ystar, h_in, mu, rho, sig2);
    clear c                                   % rmpath and delete before the next copy
    assert(isequal(h_leg, h_core), ...
        'ksc_ar1_mean: h differs from legacy %s', copies{ii});
    assert(isequal(S_leg, S_core), ...
        'ksc_ar1_mean: indicators differ from legacy %s', copies{ii});
end
assert(all(ismember(S_core, 1:7)), 'ksc_ar1_mean: indicators outside 1..7');
end

function cleanup_tmp(tmp)
rmpath(tmp);
rmdir(tmp, 's');
end
