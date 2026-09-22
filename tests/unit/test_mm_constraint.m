function test_mm_constraint
% bvar.util.mm_constraint must write one row per observed quarterly value, with the
% Mariano-Murasawa weights on the five monthly values of its own series, and must leave
% out a quarterly value whose window starts before the first month.
T = 12; n = 3;
X = reshape(1:T*n, n, T)';                          % monthly columns 1 and 3
X(:,2) = NaN;
X([3 6 9 12], 2) = [0.3; 0.6; -0.2; 0.9];          % quarterly growth, last month of each quarter
[M, z, Y, dropped] = bvar.util.mm_constraint(X, [false true false]);

assert(isequal(size(M), [3 T*n]) && isequal(z, [0.6; -0.2; 0.9]), ...
    'mm_constraint: expected three rows, for the quarters ending in months 6, 9 and 12');
assert(isequal(find(dropped), 3 + T), 'mm_constraint: the value in month 3 must be dropped');
assert(all(isnan(Y(:,2))) && isequal(Y(:,[1 3]), X(:,[1 3])), ...
    'mm_constraint: Y must blank the quarterly column and keep the monthly ones');

% the row for month 9 acting on a path gives the weighted sum of months 5 to 9
y = randn(T*n, 1);
Yp = reshape(y, n, T)';
assert(abs(M(2,:)*y - [1 2 3 2 1]/3*Yp(9:-1:5, 2)) < 1e-14, ...
    'mm_constraint: the weights are not those of Mariano and Murasawa (2003)');

% other weights, for a two-month sum
[M2, z2] = bvar.util.mm_constraint(X, [false true false], 'weights', [1 1]);
assert(size(M2,1) == 4 && isequal(z2, X([3 6 9 12], 2)), 'mm_constraint: two-month rows expected');
assert(abs(M2(1,:)*y - (Yp(3,2) + Yp(2,2))) < 1e-14, 'mm_constraint: w(1) must act on month t');

bad = { @() bvar.util.mm_constraint(X, [false false false]), 'badIsq'; ...
        @() bvar.util.mm_constraint(X, [true false]), 'badIsq'; ...
        @() bvar.util.mm_constraint(X, [false true false], 'zzz', 1), 'badOption'};
for ib = 1:size(bad, 1)
    try
        bad{ib,1}();
        error('the call with a %s input should have errored', bad{ib,2});
    catch err
        assert(strcmp(err.identifier, ['bvar:util:mm_constraint:' bad{ib,2}]), ...
            'wrong identifier: %s', err.identifier);
    end
end
fprintf('test_mm_constraint passed\n');
end
