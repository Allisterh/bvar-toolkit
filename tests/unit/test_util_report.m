function test_util_report
% bvar.util.report: the csv reads back as the table it was given, the mat carries
% the table and the settings, the stamped fields are added, a warning raised
% before the call is recorded, and the default destination is tempdir.
tmpdir = tempname; mkdir(tmpdir);
c = onCleanup(@() rmdir(tmpdir, 's'));   %#ok<NASGU>

tbl = table(["VAR-SV"; "VAR-SVO"], [-1018.7; -1008.8], [0.33; 0.31], [1; 1], ...
    'VariableNames', {'model', 'log_ML', 'nse', 'seed'});
meta = struct('file', 'macro5_Q.csv', 'T', 208, 'kappa', [0.04 0.0016]);

lastwarn('');
f = bvar.util.report('t_report', tbl, meta, tmpdir);
assert(numel(f) == 2 && exist(f{1}, 'file') == 2 && exist(f{2}, 'file') == 2, ...
    'both files should exist');

back = readtable(f{1}, 'TextType', 'string');
assert(isequal(back.Properties.VariableNames, tbl.Properties.VariableNames), ...
    'csv: column names differ');
assert(isequal(back.model, tbl.model), 'csv: text column differs');
assert(max(abs(back.log_ML - tbl.log_ML)) < 5e-5, 'csv: numbers differ');

s = load(f{2});
assert(isequal(s.tbl, tbl), 'mat: the table differs');
assert(isequal(s.meta.file, meta.file) && isequal(s.meta.kappa, meta.kappa), ...
    'mat: the settings differ');
assert(all(isfield(s.meta, {'run_time', 'matlab_version', 'last_warning', 'last_warning_id'})), ...
    'mat: the stamped fields are missing');
assert(isempty(s.meta.last_warning), 'mat: no warning was raised, none should be recorded');

    % a warning raised before the call is carried into the report
ws = warning('off', 'bvar:test:report');
cw = onCleanup(@() warning(ws));   %#ok<NASGU>
lastwarn('');
warning('bvar:test:report', 'a warning to record');
fw = bvar.util.report('t_warned', tbl, meta, tmpdir);
s = load(fw{2});
assert(strcmp(s.meta.last_warning, 'a warning to record') && ...
    strcmp(s.meta.last_warning_id, 'bvar:test:report'), 'mat: the warning is not recorded');

    % a struct of tables: one csv per field, and the struct itself in the mat
lastwarn('');
two = struct('models', tbl, 'outliers', table([1; 2], [0.98; 0.31], ...
    'VariableNames', {'period', 'probability'}));
f = bvar.util.report('t_two', two, meta, tmpdir);
assert(numel(f) == 3, 'two tables should give two csv files and one mat');
assert(strcmp(f{1}, fullfile(tmpdir, 't_two_models.csv')) && ...
    strcmp(f{2}, fullfile(tmpdir, 't_two_outliers.csv')), 'the csv names should carry the field');
back = readtable(f{2});
assert(isequal(back.period, [1; 2]), 'csv: the second table differs');
s = load(f{3});
assert(isequal(s.tbl.models, tbl) && isequal(s.tbl.outliers, two.outliers), ...
    'mat: the struct of tables differs');

    % the default destination is tempdir
lastwarn('');
f = bvar.util.report('t_default', tbl, meta);
d = onCleanup(@() delete(f{:}));   %#ok<NASGU>
assert(strncmp(f{1}, tempdir, numel(tempdir)), 'the default destination should be tempdir');

    % argument checks
bad = { @() bvar.util.report('t', struct('a', 1), meta, tmpdir), 'badTable'; ...
        @() bvar.util.report('t', struct([]), meta, tmpdir),     'badTable'; ...
        @() bvar.util.report('t', 7, meta, tmpdir),              'badTable'; ...
        @() bvar.util.report('t', tbl, 7, tmpdir),               'badMeta'};
for ib = 1:size(bad, 1)
    try
        bad{ib,1}();
        error('call %d should have errored (%s)', ib, bad{ib,2});
    catch err
        assert(strcmp(err.identifier, ['bvar:util:report:' bad{ib,2}]), ...
            'call %d gave %s, expected %s', ib, err.identifier, bad{ib,2});
    end
end
end
