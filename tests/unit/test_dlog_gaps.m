function test_dlog_gaps
% bvar.util.dlog_gaps must return the growth rates of each series and one restriction per
% interior gap, whose value is the log change over the gap; gaps at the ends of the
% sample must leave no restriction.
Lv = [100 50; 101 NaN; NaN 51; NaN 52; 104 53; 105 NaN];
[G, M, z] = bvar.util.dlog_gaps(Lv);

assert(isequal(size(G), [5 2]), 'dlog_gaps: G must have T-1 rows');
assert(abs(G(1,1) - 100*log(101/100)) < 1e-12 && all(isnan(G(2:3,1))) && ...
    abs(G(5,1) - 100*log(105/104)) < 1e-12, 'dlog_gaps: wrong growth rates of series 1');
assert(isequal(size(M), [2 10]), 'dlog_gaps: one restriction for each series, the trailing gap left out');
assert(abs(z(1) - 100*log(104/101)) < 1e-12 && abs(z(2) - 100*log(51/50)) < 1e-12, ...
    'dlog_gaps: the restriction values must be the log changes over the gaps');

% a completed path that agrees with the levels satisfies the restrictions
Gfull = 100*diff(log([100 50; 101 50.5; 102 51; 103 52; 104 53; 105 54]));
g = reshape(Gfull', [], 1);
assert(norm(M*g - z, inf) < 1e-12, 'dlog_gaps: a path consistent with the levels breaks M*g = z');
assert(isequal(find(M(1,:)), [3 5 7]) && isequal(find(M(2,:)), [2 4]), ...
    'dlog_gaps: the restrictions must act on the growth rates across each gap');

try
    bvar.util.dlog_gaps([1; -1]);
    error('negative levels should have errored');
catch err
    assert(strcmp(err.identifier, 'bvar:util:dlog_gaps:badData'), 'wrong identifier: %s', err.identifier);
end
fprintf('test_dlog_gaps passed\n');
end
