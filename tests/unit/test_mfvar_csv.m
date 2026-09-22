function test_mfvar_csv
% bvar.models.mfvar_csv must keep the restrictions in every draw, return the data where
% they are observed, repeat itself under one seed, and refuse the inputs its header rules
% out. The draw of the missing values is checked against dense conditioning in
% test_missing_var, and the h update against a grid in test_csv_armh_block.
rng(20260921, 'twister');
n = 3; p = 2; T = 62;
A = [0.1 0.1 0.2; 0.4 0.1 0.1; 0.1 0.3 0.1; 0.3 0.3 0.5; 0.1 0 0; 0 0.1 0; 0 0 0.1];
Yt = zeros(T, n);
for t = p+1:T
    x = [1, reshape(Yt(t-1:-1:t-p,:)', 1, n*p)];
    Yt(t,:) = x*A + 0.5*randn(1,n);
end
X = Yt;  X(:,3) = NaN;
qend = (5:3:T)';  y3 = Yt(:,3);
X(qend,3) = y3(qend - (0:4))*[1 2 3 2 1]'/3;
[M, z, Y] = bvar.util.mm_constraint(X, [false false true]);
Y(20,1) = NaN;                                     % a hole in a monthly series

opts = {'M', M, 'z', z, 'sig2', [NaN NaN .1], 'nsim', 60, 'burnin', 30, 'seed', 5, 'draws', true};
r1 = bvar.models.mfvar_csv(Y, p, opts{:});
r2 = bvar.models.mfvar_csv(Y, p, opts{:});

assert(isequal(r1.Y_mean, r2.Y_mean) && isequal(r1.draws.h, r2.draws.h), ...
    'mfvar_csv: two runs under one seed must give the same draws');
assert(r1.max_resid < 1e-9, 'mfvar_csv: a draw breaks M*y = z (%.2e)', r1.max_resid);
obs = ~isnan(Y);
for j = 1:size(r1.Y_q, 3)
    q = r1.Y_q(:,:,j);
    assert(isequal(q(obs), Y(obs)), 'mfvar_csv: the quantiles must equal the data where observed');
end
assert(isequal(r1.Y_mean(obs), Y(obs)), 'mfvar_csv: Y_mean must equal the data where observed');
assert(all(r1.Y_q(:,:,1) <= r1.Y_q(:,:,end), 'all'), 'mfvar_csv: the quantiles must be ordered');
assert(isequal(size(r1.A_mean), [1+n*p n]) && numel(r1.h_mean) == T-p ...
    && isequal(size(r1.draws.ym), [60 sum(~obs(:))]), 'mfvar_csv: wrong output sizes');
assert(abs(r1.sig2(3) - .1) < eps && all(r1.sig2(1:2) > 0), ...
    'mfvar_csv: a given sig2 must be kept and a NaN one estimated');

% the prior variances, in the regressor order of bvar.util.build_lags: with sig2 = [1 4 9],
% kappa = [.04 .01] and p = 2, entry by entry from the formulas of Chan, Poon and Zhu (2023)
r3 = bvar.models.mfvar_csv(Y, p, 'M', M, 'z', z, 'sig2', [1 4 9], 'nsim', 2, 'burnin', 0, 'seed', 1);
V = r3.prior.V;
assert(isequal(size(V), [1+n*p n]), 'mfvar_csv: prior.V must be k x n');
chk = [1 3 900;                 % intercept of equation 3: 100*s_3^2
       2 1 .04;                 % lag 1 of variable 1 in equation 1: kappa_1
       3 1 .01*1/4;             % lag 1 of variable 2 in equation 1: kappa_2*s_1^2/s_2^2
       4 2 .01*4/9;             % lag 1 of variable 3 in equation 2
       5 3 .01*9/(4*1);         % lag 2 of variable 1 in equation 3: kappa_2*s_3^2/(4*s_1^2)
       6 2 .04/4];              % lag 2 of variable 2 in equation 2: kappa_1/4
for r = 1:size(chk,1)
    assert(abs(V(chk(r,1), chk(r,2)) - chk(r,3)) < 1e-12, ...
        'mfvar_csv: prior variance in row %d, equation %d is %g, expected %g', ...
        chk(r,1), chk(r,2), V(chk(r,1), chk(r,2)), chk(r,3));
end

bad = { @() bvar.models.mfvar_csv(Y, p, 'M', M, 'z', z), 'badOption', 'no sig2 for the unobserved series'; ...
        @() bvar.models.mfvar_csv(Y, p, 'sig2', [1 1 1], 'zzz', 1), 'badOption', 'an unknown option'; ...
        @() bvar.models.mfvar_csv(Y, p, 'sig2', [1 1 1], 'nsim', 0), 'badOption', 'a zero nsim'; ...
        @() bvar.models.mfvar_csv([Y(:,1:2); Inf Inf], p), 'badData', 'an Inf'; ...
        @() bvar.models.mfvar_csv(Y(:,1), p), 'badData', 'one series'};
for ib = 1:size(bad, 1)
    try
        bad{ib,1}();
        error('%s should have errored', bad{ib,3});
    catch err
        assert(strcmp(err.identifier, ['bvar:models:mfvar_csv:' bad{ib,2}]), ...
            'wrong identifier for %s: %s', bad{ib,3}, err.identifier);
    end
end
fprintf('test_mfvar_csv passed\n');
end
