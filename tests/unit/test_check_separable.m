function test_check_separable
% bvar.structural.check_separable must separate a pair of shocks by their signs
% or by a ranking restriction, must report the pairs it cannot separate, and must
% ignore the scale of a ranking row. On the 35-variable restriction set of the
% SVAR-sign package every pair must come out separated, and five pairs
% unseparated when the ranking restrictions are withheld.

% --- the paper's own restrictions, read from the driver that states them ---
root = getappdata(0, 'bvar_repo_root');
pkg = fullfile(root, 'replications', 'chan_matthes_yu2026_qe_svarsign', 'legacy');
src = fileread(fullfile(pkg, 'main_35VAR_Figs2to10_except5.m'));
i0 = strfind(src, 'demand = [');
i1 = strfind(src, 'start_time = clock');
assert(isscalar(i0) && isscalar(i1), 'the restriction block was not found in the driver');
n = 35;                                                      %#ok<NASGU>
eval(regexprep(src(i0:i1-1), 'end\s*$', ''));                % sets S, Rineq, m, k

[ok, bad] = bvar.structural.check_separable(S, Rineq);
assert(ok && isempty(bad), 'the paper''s restrictions should separate every pair');
[ok0, bad0] = bvar.structural.check_separable(S);
assert(~ok0 && isequal(bad0, [1 2; 1 3; 1 5; 2 5; 3 5]), ...
    'the sign restrictions alone should leave exactly those five pairs unseparated');

% --- separated by signs: same sign on variable 1, opposite on variable 2 ---
S2 = [1 1; 1 -1];
assert(bvar.structural.check_separable(S2), 'a same-and-opposite pair is separated');

% --- the same sign twice, and nothing opposite, is not enough ---
S3 = [1 1; 1 1];
[ok3, bad3] = bvar.structural.check_separable(S3);
assert(~ok3 && isequal(bad3, [1 2]), 'a pair agreeing on every sign is not separated');

% --- a ranking restriction pointing opposite ways separates that pair ---
R = zeros(2, 2, 1);
R(1,:,1) = [1 -1];        % shock 1: response 1 at most response 2
R(2,:,1) = [-1 1];        % shock 2: the other way
assert(bvar.structural.check_separable(S3, R), ...
    'opposite rankings on the same two variables separate the pair');
R(2,:,1) = [1 -1];        % the same way for both separates nothing
assert(~bvar.structural.check_separable(S3, R), ...
    'identical rankings do not separate the pair');

% --- scale does not matter, only the direction ---
R(1,:,1) = [2 -2];  R(2,:,1) = [-1 1];
assert(bvar.structural.check_separable(S3, R), 'the test must be free of scale');

% --- the shape guard ---
try
    bvar.structural.check_separable(S3, zeros(3, 2));
    error('test_check_separable:noGuard', 'a wrong-shaped Rineq must raise');
catch err
    assert(strcmp(err.identifier, 'bvar:structural:check_separable:badRineq'), ...
        'the wrong error was raised for a bad Rineq');
end
end
