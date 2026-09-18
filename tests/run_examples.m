% run_examples - run setup.m, then every script in examples/ and the own-data
% script of each tutorial (tutorials/*/your_data.m), each in its own workspace;
% error if any of them fails.
% Usage (from repo root or anywhere):  matlab -batch "run('tests/run_examples.m')"
%
% It checks that setup.m puts the toolkit on the path, as the README quick start
% assumes, and that every example runs to the end without an error. Scripts are
% found by name (examples/ex*.m, tutorials/*/your_data.m), so a new one is
% covered as soon as it is added.
% The CI workflow runs this after the unit suite (see
% .github/workflows/unit-tests.yml).

function run_examples

thisdir = fileparts(mfilename('fullpath'));
root = fileparts(thisdir);

% No example needs the System Identification Toolbox, which only one replication
% driver uses, so setup.m's warning about it is turned off while this runs.
% Otherwise it would repeat once per example, since each example runs setup.m.
wident = warning('off', 'bvar:setup:noIdent');
restore = onCleanup(@() warning(wident)); %#ok<NASGU>

% Take core/ and third_party/ off the path first, so the check below depends on
% setup.m alone even in a session where they were already added.
w = warning('off', 'MATLAB:rmpath:DirNotFound');
rmpath(fullfile(root, 'core'), fullfile(root, 'third_party'));
warning(w);
run(fullfile(root, 'setup.m'));
if isempty(which('bvar.util.build_lags'))
    error('setup.m ran, but bvar.util.build_lags is not on the path');
end
fprintf('PASS  setup.m\n');

exdir = fullfile(root, 'examples');
ex = dir(fullfile(exdir, 'ex*.m'));
names = sort(erase({ex.name}, '.m'));
if isempty(names)
    error('no examples found in %s', exdir);
end
files = fullfile(exdir, strcat(names, '.m'));
n_ex = numel(names);
tut = dir(fullfile(root, 'tutorials', '*', 'your_data.m'));
for ii = 1:numel(tut)
    [~, folder] = fileparts(tut(ii).folder);
    names{end+1} = ['tutorials/' folder '/your_data']; %#ok<AGROW>
    files{end+1} = fullfile(tut(ii).folder, tut(ii).name); %#ok<AGROW>
end

failed = {};
for ii = 1:numel(names)
    fprintf('\n---- %s ----\n', names{ii});
    t0 = tic;
    try
        run_one(files{ii});
        fprintf('PASS  %s (%.0f s)\n', names{ii}, toc(t0));
    catch err
        failed{end+1} = names{ii}; %#ok<AGROW>
        fprintf('FAIL  %s\n%s\n', names{ii}, getReport(err, 'extended', 'hyperlinks', 'off'));
    end
    close all force
end

if ~isempty(failed)
    error('%d of %d scripts failed: %s', numel(failed), numel(names), strjoin(failed, ', '));
end
n_tut = numel(names) - n_ex;
fprintf('\nAll %d examples and %d tutorial script%s ran.\n', n_ex, n_tut, repmat('s', 1, n_tut ~= 1));
end

function run_one(file)
% Each example runs in this function's workspace, so no example can see or clear
% the variables of another, or those of the loop above.
run(file);
end
