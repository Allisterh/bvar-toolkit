function test_mlvarsv_options
% the 'data' and 'r' options of chan2023_joe_mlvarsv/run_all and run_ml. The
% package's own data passed as 'data', with 'r' = 2, reproduce the default calls
% bitwise for all five models (every output, and the terminal rng state); 'r'
% sets the number of VAR-FSV factors; run_ml passes both options to run_all and
% prints the VAR-SVO comparison with Table 6 of the paper only on the package's
% data.
root = getappdata(0, 'bvar_repo_root');
repdir = fullfile(root, 'replications', 'chan2023_joe_mlvarsv');
addpath(repdir); cp = onCleanup(@() rmpath(repdir));
resolved = which('run_all');
assert(strncmpi(resolved, repdir, numel(repdir)), ...
    'run_all must resolve from the ml_varsv package, got %s', resolved);
od = cd(repdir); guard = onCleanup(@() cd(od));
pr = preset();
clear guard

nsim = 20; burnin = 5; seed = 20260919; M = 100;
varid = [1,22,59,120];      % n = 4: the first four of the commented n = 7 selection
data_all = load(fullfile(repdir, 'legacy', pr.data_file));
D = data_all(:, varid);

    % the package's data as 'data' reproduces the default call
for m = 1:5
    a = run_all(m, [], [], nsim, burnin, seed, varid);
    sa = rng;
    b = run_all(m, [], [], nsim, burnin, seed, [], 'data', D, 'r', pr.r);
    sb = rng;
    assert(a.is_package_data && ~b.is_package_data && isempty(b.varid), ...
        'model %d: is_package_data or varid is wrong', m);
    a = rmfield(a, {'varid', 'is_package_data'});
    b = rmfield(b, {'varid', 'is_package_data'});
    assert(isequaln(a, b), 'model %d: the data option changes the output', m);
    assert(isequal(sa, sb), 'model %d: the terminal rng state differs', m);
end

    % 'r' sets the number of factors
for r = [1 3]
    c = run_all(4, [], [], nsim, burnin, seed, [], 'data', D, 'r', r);
    n = size(D, 2);
    assert(c.r == r && size(c.store_F, 3) == r && size(c.store_h, 3) == n+r ...
        && size(c.store_l, 2) == n*r-r*(r+1)/2, 'r = %d: wrong sizes', r);
end

    % run_ml passes the options on; the Table 6 notice needs the package's data
a = run_ml(4, [], [], nsim, burnin, seed, varid, 'M', M);
b = run_ml(4, [], [], nsim, burnin, seed, [], 'M', M, 'data', D, 'r', pr.r);
assert(isequal(a.lml, b.lml) && isequal(a.lmlstd, b.lmlstd) && isequaln(a.ml, b.ml), ...
    'VAR-FSV: run_ml with the data option differs');
ta = evalc('a = run_ml(5, [], [], nsim, burnin, seed, varid, ''M'', M);');
tb = evalc('b = run_ml(5, [], [], nsim, burnin, seed, [], ''M'', M, ''data'', D);');
tc = evalc('c = run_ml(5, [], [], nsim, burnin, seed, [], ''M'', M, ''data'', D, ''bugcompat'', true);');
assert(isequal(a.lml, b.lml) && isequal(a.lmlstd, b.lmlstd) && isequaln(a.ml, b.ml), ...
    'VAR-SVO: run_ml with the data option differs');
assert(contains(ta, 'Table 6') && ~contains(tb, 'Table 6') && ~contains(tc, 'Table 6'), ...
    'the Table 6 notice must print on the package''s data only');
assert(contains(tc, 'bugcompat is on') && ~isequal(c.lml, b.lml), ...
    'the bugcompat notice and computation must not depend on the data');

    % invalid options
bad = {{'datta', D}, 'run_all:badOption'; {'data'}, 'run_all:badOption'; ...
    {'r', 0}, 'run_all:badOption'; {'r', 1.5}, 'run_all:badOption'; ...
    {'data', D(1:pr.n0, :)}, 'run_all:badData'; {'data', D(:, 1)}, 'run_all:badData'; ...
    {'data', [D(1:end-1, :); NaN(1, size(D, 2))]}, 'run_all:badData'};
for ib = 1:size(bad, 1)
    expect_error(@() run_all(2, [], [], 2, 1, seed, [], bad{ib, 1}{:}), bad{ib, 2});
end
expect_error(@() run_ml(2, [], [], 2, 1, seed, [], 'datta', D), 'run_ml:badOption');
end

% -------------------------------------------------------------------------
function expect_error(f, id)
try
    f();
catch err
    assert(strcmp(err.identifier, id), 'expected %s, got %s: %s', id, err.identifier, err.message);
    return
end
error('expected the error %s', id);
end
