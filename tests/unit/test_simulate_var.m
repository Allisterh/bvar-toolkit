function test_simulate_var
% simulate_var against an inline spelling of the same draws, its stationary moments, the start
% from given initial values, and its errors
n = 2; p = 2;
A = [.5 1; .6 .1; .2 .3; -.2 0; .1 .2];   % intercepts, lag 1, lag 2 (k x n)
Sig = [1 .3; .3 .5];
F = [A(2:end, :)'; eye(n), zeros(n)];
assert(max(abs(eig(F))) < 1, 'test_simulate_var: the test VAR must be stable');
c = [A(1, :)'; zeros(n, 1)];
mu = (eye(n*p) - F)\c;
Q = blkdiag(Sig, zeros(n));
V = reshape((eye((n*p)^2) - kron(F, F))\Q(:), n*p, n*p);   % vec(V) = (I - F kron F)^{-1} vec(Q)

% draw for draw against an inline loop on the same random numbers
rng(11, 'twister');
[Y, Y0] = bvar.util.simulate_var(A, Sig, 50);
rng(11, 'twister');
s = mu + chol((V + V')/2, 'lower')*randn(n*p, 1);
Yi = zeros(p + 50, n);
Yi(1:p, :) = [s(3:4)'; s(1:2)'];          % y_{-1}, y_0
C = chol(Sig, 'lower');
for t = p+1:p+50
    Yi(t, :) = A(1, :) + Yi(t-1, :)*A(2:3, :) + Yi(t-2, :)*A(4:5, :) + (C*randn(n, 1))';
end
assert(max(abs([Y0; Y] - Yi), [], 'all') < 1e-12, 'simulate_var: differs from the inline draws');

% the stationary mean and covariance, from a long sample
rng(12, 'twister');
Ylong = bvar.util.simulate_var(A, Sig, 400000);
assert(max(abs(mean(Ylong)' - mu(1:n))) < .02, 'simulate_var: wrong stationary mean');
assert(max(abs(cov(Ylong) - V(1:n, 1:n)), [], 'all') < .03*max(abs(V(:))), ...
    'simulate_var: wrong stationary covariance');

% given initial values: no draws for them, the same recursion after
rng(13, 'twister');
Ys = bvar.util.simulate_var(A, Sig, 10, [1 2; 3 4]);
rng(13, 'twister');
Yj = [1 2; 3 4; zeros(10, n)];
for t = 3:12
    Yj(t, :) = A(1, :) + Yj(t-1, :)*A(2:3, :) + Yj(t-2, :)*A(4:5, :) + (C*randn(n, 1))';
end
assert(max(abs(Ys - Yj(3:end, :)), [], 'all') < 1e-12, 'simulate_var: wrong path from given Y0');

% errors: an unstable VAR without initial values, A of the wrong size, Y0 of the wrong size
Au = A; Au(2, 1) = 1.2;
assert_error(@() bvar.util.simulate_var(Au, Sig, 5), 'simulate_var:unstable');
bvar.util.simulate_var(Au, Sig, 5, zeros(p, n));                  % runs from given values
assert_error(@() bvar.util.simulate_var(A(1:4, :), Sig, 5), 'simulate_var:badA');
assert_error(@() bvar.util.simulate_var(A, Sig, 5, zeros(1, n)), 'simulate_var:badY0');
end

function assert_error(f, id)
try
    f();
catch err
    assert(strcmp(err.identifier, id), 'expected %s, got %s', id, err.identifier);
    return
end
error('test_simulate_var:noError', 'expected the error %s', id);
end
