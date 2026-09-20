% bvar.util.report - write a comparison table to <outdir>/<name>.csv and the same
% table with a settings struct to <outdir>/<name>.mat, and print the paths. outdir
% defaults to tempdir, so a script that calls this leaves the working tree and a
% CI checkout clean.
%
%   f = bvar.util.report(name, tbl, meta)
%   f = bvar.util.report(name, tbl, meta, outdir)
%
%   name : base name of the files
%   tbl  : one table, or a struct of tables when the run produces more than one
%          shape of result, written one csv per field as <name>_<field>.csv.
%          Settings that differ by row belong in the columns; a csv holds its
%          table alone, so that reading it needs no convention beyond readtable.
%   meta : struct describing the run (data file, sample, seeds, chain lengths,
%          hyperparameters). It gains the run time, the MATLAB version and the
%          last warning raised, which is the caller's to clear with lastwarn('')
%          before the work starts.
%   f    : the paths written, the csv files first and the mat last
%
% load() on the .mat returns the two variables tbl and meta.

function f = report(name, tbl, meta, outdir)
if nargin < 4 || isempty(outdir), outdir = tempdir; end
assert(istable(tbl) || (isstruct(tbl) && isscalar(tbl)), 'bvar:util:report:badTable', ...
    'tbl must be a table or a scalar struct of tables');
if isstruct(tbl)
    fn = fieldnames(tbl);
    assert(~isempty(fn) && all(cellfun(@(x) istable(tbl.(x)), fn)), ...
        'bvar:util:report:badTable', 'every field of tbl must be a table');
end
assert(isstruct(meta) && isscalar(meta), 'bvar:util:report:badMeta', ...
    'meta must be a scalar struct');
if ~exist(outdir, 'dir'), mkdir(outdir); end

[wmsg, wid] = lastwarn;
meta.run_time = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm'));
meta.matlab_version = version;
meta.last_warning = wmsg;
meta.last_warning_id = wid;

if istable(tbl)
    f = {fullfile(outdir, [name '.csv'])};
    writetable(tbl, f{1});
else
    f = cell(1, numel(fn));
    for i = 1:numel(fn)
        f{i} = fullfile(outdir, [name '_' fn{i} '.csv']);
        writetable(tbl.(fn{i}), f{i});
    end
end
f{end+1} = fullfile(outdir, [name '.mat']);
save(f{end}, 'tbl', 'meta');
fprintf('\nreport written\n');
fprintf('  %s\n', f{1:end-1});
fprintf('  %s  (load it for the tables and the settings)\n', f{end});
if ~isempty(wmsg)
    fprintf('  the run raised at least one warning, the last of them recorded in meta:\n  %s\n', wmsg);
end
end
