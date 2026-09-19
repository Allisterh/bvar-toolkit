function test_ksc_rw_h0
% bvar.sv.ksc_rw_h0 must reproduce, draw-for-draw under one seed, EVERY legacy
% copy it canonicalizes: springer/kronecker/mahp/ml_tvpsv SVRW.m and HYB
% sample_SVRW.m, each run from a tempdir copy with the one-factor substitution
% of one_factor_patch.
root = getappdata(0, 'bvar_repo_root');

rng(41, 'twister');                          % fixed test data
T = 137;
e = randn(T,1).*exp(0.3*randn(T,1));
ystar = log(e.^2 + 1e-4);
h_in = 0.2*randn(T,1);
sig2 = 0.12;                                 % state innovation VARIANCE
h0 = 0.4;

rng(7, 'twister');
h_core = bvar.sv.ksc_rw_h0(ystar, h_in, sig2, h0);

copies = { ...
    'chan2020_springer_largebvar/legacy/SVRW.m'; ...
    'chan2020_jbes_kronecker/legacy/realtime_forecasts/SVRW.m'; ...
    'chan2021_ijf_mahp/legacy/SVRW.m'; ...
    'chan2023_jbes_hybtvp/legacy/utility/sample_SVRW.m'; ...
    'chan_eisenstat2018_jae_mltvpsv/legacy/SVRW.m'};

for ii = 1:numel(copies)
    [~, fn] = fileparts(copies{ii});
    tmp = tempname; mkdir(tmp);
    copyfile(fullfile(root, 'replications', copies{ii}), tmp);
    one_factor_patch(fullfile(tmp, [fn '.m']), copies{ii});
    addpath(tmp); c = onCleanup(@() cleanup_tmp(tmp));  % prepended -> patched copy shadows
    rng(7, 'twister');
    h_leg = feval(fn, ystar, h_in, sig2, h0);
    clear c                                   % rmpath and delete before the next copy
    assert(isequal(h_leg, h_core), 'ksc_rw_h0: differs from legacy %s', copies{ii});
end
assert(all(isfinite(h_core)), 'ksc_rw_h0: non-finite log-volatility draw');
end

function cleanup_tmp(tmp)
rmpath(tmp);
rmdir(tmp, 's');
end
