function test_select_obs
% bvar.util.select_obs must reproduce the two illustrations of Chan, Poon and Zhu (2023),
% Section 2.1, as ssm.select_obs does in statespace-toolkit, and [So, Sm] must be a
% permutation that splits and rebuilds the stacked data.

% the paper's first illustration: T = 2, n = 3, with y_{3,1}, y_{1,2} and y_{3,2} missing
Y = [1 2 NaN; NaN 5 NaN];
[So, Sm, yo] = bvar.util.select_obs(Y);
assert(isequal(full(So), [1 0 0; 0 1 0; 0 0 0; 0 0 0; 0 0 1; 0 0 0]), ...
    'select_obs: So differs from the illustration in the paper');
assert(isequal(full(Sm), [0 0 0; 0 0 0; 1 0 0; 0 1 0; 0 0 0; 0 0 1]), ...
    'select_obs: Sm differs from the illustration in the paper');
assert(isequal(yo, [1; 2; 5]), 'select_obs: yo must be the observed values in stacked order');

% the paper's second illustration: y_{1,t} observed every third period, the rest always
T = 9; n = 4;
Y = randn(T, n);
Y(mod(1:T,3) ~= 0, 1) = NaN;
[So, Sm] = bvar.util.select_obs(Y);
bo = cell(T,1); bm = cell(T,1);
for t = 1:T
    if mod(t,3) == 0
        bo{t} = eye(n);            bm{t} = zeros(n,0);
    else
        bo{t} = [zeros(1,n-1); eye(n-1)];   bm{t} = [1; zeros(n-1,1)];
    end
end
assert(isequal(full(So), blkdiag(bo{:})), 'select_obs: So is not the block diagonal of the paper');
assert(isequal(full(Sm), blkdiag(bm{:})), 'select_obs: Sm is not the block diagonal of the paper');

% [So, Sm] is a permutation matrix, and it splits and rebuilds a complete path
P = [So Sm];
assert(all(sum(P,1) == 1) && all(sum(P,2) == 1) && nnz(P) == T*n, ...
    'select_obs: [So Sm] must be a permutation matrix');
y = randn(T*n, 1);
assert(isequal(So*(So'*y) + Sm*(Sm'*y), y), 'select_obs: y = So*yo + Sm*ym must hold');

% patterns with nothing missing, or nothing observed, and a bad input
[So, Sm, yo] = bvar.util.select_obs(ones(5,2));
assert(isequal(size(So), [10 10]) && isequal(size(Sm), [10 0]) && numel(yo) == 10, ...
    'select_obs: a complete panel must give an empty Sm');
[So, Sm, yo] = bvar.util.select_obs(nan(5,2));
assert(isequal(size(So), [10 0]) && isequal(size(Sm), [10 10]) && isempty(yo), ...
    'select_obs: an empty panel must give an empty So');
try
    bvar.util.select_obs('abc');
    error('a character input should have errored');
catch err
    assert(strcmp(err.identifier, 'bvar:util:select_obs:badY'), 'wrong identifier: %s', err.identifier);
end
fprintf('test_select_obs passed\n');
end
