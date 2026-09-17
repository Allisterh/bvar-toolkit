function test_run_jointden
% equivalence of the functionized marginal-likelihood surface
% (replications/chan2022_qe_acp/run_jointden.m) with the legacy script
% main_ACP_jointden.m, run from a tempdir copy on a coarser grid of the two
% shrinkage hyperparameters. Asserts isequal on the grid, on the log marginal
% likelihood at every grid point, on the normalized surface, and on the
% symmetric-prior optimum and its log marginal likelihood.
%
% The grid is kappa1 = 0.01:.01:.2 by kappa2 = .001:.001:0.012, 20 x 12 points
% with the endpoints of the paper's 191 x 56 grid. It is not square, so a
% transposed index fails the test.
%
% PATCHES TO THE LEGACY SCRIPT, each asserted to match exactly once:
%   1. `clear; clc;` removed - run from a function it would wipe the harness's
%      own bookkeeping;
%   2. the grid line, with the two steps widened to .01 and .001;
%   3. the figure block at the end removed, since it opens windows and is not
%      part of the computation.
% The grid patch changes only the two step sizes, and the other two remove lines
% that compute nothing.
%
% Two settings in preset.m are out of reach of the equivalence run: the default
% grid, which the run replaces, and the subjective-prior point, which only the
% figure uses. Both are checked against the legacy text. Every other setting
% run_jointden reads from preset.m enters the computation: the data file, the 15
% variables, the 8 initial observations, the lag length, kappa3 and kappa4, and
% the variables in levels.
%
% The legacy script's run on the paper's full grid is recorded in
% tests/golden/chan2022_qe_acp/main_ACP_jointden_savegolden_20260917_0835/.
root = getappdata(0, 'bvar_repo_root');
leg = fullfile(root, 'replications', 'chan2022_qe_acp', 'legacy');
repdir = fullfile(root, 'replications', 'chan2022_qe_acp');

k1 = 0.01:.01:.2;
k2 = .001:.001:0.012;
grid_legacy = '[Kappa1,Kappa2] = meshgrid(0.01:.001:.2,.001:.0002:0.012);';
grid_test = '[Kappa1,Kappa2] = meshgrid(0.01:.01:.2,.001:.001:0.012);';
marker_legacy = 'scatter(.04,.0016,''s'',''filled'');';

txt = fileread(fullfile(leg, 'main_ACP_jointden.m'));

% --- the two settings the equivalence run cannot reach ---
od = cd(repdir);
guard = onCleanup(@() cd(od));
pr = preset();
clear guard
assert(numel(strfind(txt, grid_legacy)) == 1, 'expected exactly one legacy grid line');
assert(isequal(pr.jd.kappa1_grid, 0.01:.001:.2) && isequal(pr.jd.kappa2_grid, .001:.0002:0.012), ...
    'the default grid in preset.m differs from the legacy grid');
assert(numel(strfind(txt, marker_legacy)) == 1, 'expected exactly one subjective-prior marker');
assert(isequal(pr.jd.subjective, [.04,.0016]), ...
    'pr.jd.subjective differs from the point the legacy figure marks');

% --- tempdir: patched script, verbatim utilities, and the data file ---
tmp = tempname; mkdir(tmp); mkdir(fullfile(tmp, 'utility'));
ctmp = onCleanup(@() cleanup_tmp(tmp));
u = dir(fullfile(leg, 'utility', '*.m'));
for k = 1:numel(u)
    copyfile(fullfile(leg, 'utility', u(k).name), fullfile(tmp, 'utility', u(k).name));
end
copyfile(fullfile(leg, 'database_2019Q4.xlsx'), fullfile(tmp, 'database_2019Q4.xlsx'));

txt = patch_once(txt, 'clear; clc;', '% [clear removed by test_run_jointden]', 'clear');
txt = patch_once(txt, grid_legacy, grid_test, 'the grid line');
cut = strfind(txt, 'figure; subplot(1,2,1);');
assert(numel(cut) == 1, 'expected exactly one figure block');
txt = txt(1:cut-1);
fid = fopen(fullfile(tmp, 'main_ACP_jointden.m'), 'w'); fwrite(fid, txt); fclose(fid);

addpath(repdir); cp = onCleanup(@() rmpath(repdir));
resolved = which('run_jointden');
assert(strncmpi(resolved, repdir, numel(repdir)), ...
    'run_jointden must resolve from the ACP package, got %s', resolved);

% the functionized driver first, before the legacy script puts its utility
% folder (and its QR.m) on the path
[~, C] = evalc('run_jointden(k1, k2)');
L = run_legacy(tmp);

assert(isequal(size(C.store_lml), [numel(k2), numel(k1)]), 'store_lml has the wrong shape');
assert(isequal(L.Kappa1, C.Kappa1) && isequal(L.Kappa2, C.Kappa2), 'the grid differs');
assert(isequal(L.store_lml, C.store_lml), 'store_lml differs');
assert(isequal(L.store_ml, C.store_ml), 'store_ml differs');
assert(isequal(L.ml_Sym, C.ml_Sym), 'ml_Sym differs');
assert(isequal(L.kappa_Sym, C.kappa_Sym), 'kappa_Sym differs');

% the surface must vary, or the comparison is vacuous
assert(all(isfinite(C.store_lml(:))) && max(C.store_lml(:)) > min(C.store_lml(:)), ...
    'store_lml is not a finite, varying surface');
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
function out = run_legacy(tmpdir_)
% Run the patched legacy script from the tempdir as the working directory; its
% addpath('./utility') and its bare data filename are both relative. Locals
% carry a trailing underscore so the script's own variables cannot clobber them.
od_ = cd(tmpdir_);

Kappa1 = []; Kappa2 = []; store_lml = []; store_ml = []; ml_Sym = []; kappa_Sym = [];

resolved_ = which('main_ACP_jointden');
assert(strncmpi(resolved_, tmpdir_, numel(tmpdir_)), ...
    'main_ACP_jointden must resolve from the tempdir copy, got %s', resolved_);

try
    evalc('main_ACP_jointden');
catch err_
    cd(od_);
    rethrow(err_);
end
cd(od_);

out = struct('Kappa1',Kappa1, 'Kappa2',Kappa2, 'store_lml',store_lml, ...
    'store_ml',store_ml, 'ml_Sym',ml_Sym, 'kappa_Sym',kappa_Sym);
end
