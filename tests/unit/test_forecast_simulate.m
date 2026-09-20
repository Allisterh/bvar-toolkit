function test_forecast_simulate
% bvar.forecast.simulate: with the volatility switched off each spec must return
% the analytic Gaussian density of the outturn at the first horizon, the three
% specs must agree when their covariances coincide, and the conditional means
% must average over simulated paths to the deterministic iteration that
% bvar.forecast.predictive returns.
rng(20260921, 'twister');
n = 3; p = 2; k = n*p + 1;
A = [0.2 -0.1 0.05; 0.5 0.1 0; 0.1 0.4 0.1; 0 0.2 0.5; 0.1 0 0; 0 0.1 0; 0 0 0.1];
CS = [1 0 0; .4 1 0; .2 .3 1]/2;
Sig = CS*CS';
ylag = [0.3 -0.2; 0.1 0.4; -0.5 0.2];
yobs = [0.2 0.1 -0.3; 0.4 -0.1 0.2];

%% ---- 1. the first horizon against the analytic density ----
cfg = struct('ylag', ylag, 'H', 2, 'yobs', yobs);
EY = [1, reshape(ylag, 1, [])]*A;
u = yobs(1,:) - EY;
want_var = diag(Sig)';
want = -.5*log(2*pi*want_var) - .5*u.^2./want_var;
wantj = -n/2*log(2*pi) - sum(log(diag(chol(Sig,'lower')))) - .5*(u/Sig*u');

[yh, ld, lj, sd0] = bvar.forecast.simulate('gauss', struct('A', A, 'Sig', Sig), cfg);
assert(max(abs(yh(1,:) - EY)) < 1e-14, 'gauss: the first conditional mean is wrong');
assert(max(max(abs(sd0 - repmat(sqrt(diag(Sig))', 2, 1)))) < 1e-14, ...
    'gauss: the conditional standard deviations must be those of Sig at every horizon');
assert(max(abs(ld(1,:) - want)) < 1e-12, 'gauss: the first log density is wrong');
assert(abs(lj(1) - wantj) < 1e-12, 'gauss: the first joint log density is wrong');

    % csv with no volatility innovation and h_T = 0 is the same model
d = struct('A', A, 'Sig', Sig, 'h_T', 0, 'phi', 0, 'sigh2', 0);
[~, ld2, lj2] = bvar.forecast.simulate('csv', d, cfg);
assert(max(abs(ld2(1,:) - want)) < 1e-12 && abs(lj2(1) - wantj) < 1e-12, ...
    'csv with a fixed volatility must match the constant-covariance density');

    % oisv with no volatility innovation, at the h_T that reproduces Sig
B0 = inv(CS);                                   % so B0\diag(1)/B0' = CS*CS' = Sig
d = struct('A', A, 'impact', B0, 'h_T', zeros(1,n), 'phi', zeros(1,n), 'sig2', zeros(1,n));
[~, ld3, lj3] = bvar.forecast.simulate('oisv', d, cfg);
assert(max(abs(ld3(1,:) - want)) < 1e-11 && abs(lj3(1) - wantj) < 1e-11, ...
    'oisv with a fixed volatility must match the constant-covariance density');

%% ---- 2. a volatility level shifts the density by the scale it implies ----
c = 0.7;
d = struct('A', A, 'Sig', Sig, 'h_T', c, 'phi', 1, 'sigh2', 0);
[~, ld4] = bvar.forecast.simulate('csv', d, cfg);
v = exp(c)*diag(Sig)';
want4 = -.5*log(2*pi*v) - .5*u.^2./v;
assert(max(abs(ld4(1,:) - want4)) < 1e-12, ...
    'csv: a constant log-volatility must scale the covariance by its exponential');

%% ---- 3. the conditional means average to the deterministic iteration ----
H = 4;
cfgm = struct('ylag', ylag, 'H', H);
R = 4000;
acc = zeros(H, n);
for r = 1:R
    acc = acc + bvar.forecast.simulate('gauss', struct('A', A, 'Sig', Sig), cfgm);
end
acc = acc/R;
mu = bvar.forecast.predictive(A(:)', reshape(Sig, 1, n, n), ylag, H);
mu = reshape(mu, n, H)';
    % the Monte Carlo error of the mean at the last horizon, from the predictive
    % standard deviation of that horizon
[~, sd] = bvar.forecast.predictive(A(:)', reshape(Sig, 1, n, n), ylag, H);
tol = 4*max(sd(:))/sqrt(R);
assert(max(abs(acc(:) - mu(:))) < tol, ...
    'the simulated conditional means do not average to the deterministic iteration');

%% ---- 4. argument checks ----
bad = { @() bvar.forecast.simulate('zzz', struct('A', A, 'Sig', Sig), cfg), 'badSpec'; ...
        @() bvar.forecast.simulate('gauss', struct('A', A(1:end-1,:), 'Sig', Sig), cfg), 'badDims'; ...
        @() bvar.forecast.simulate('gauss', struct('A', A, 'Sig', Sig), ...
            struct('ylag', ylag', 'H', 2)), 'badLags'};
for ib = 1:size(bad, 1)
    try
        bad{ib,1}();
        error('call %d should have errored (%s)', ib, bad{ib,2});
    catch err
        assert(strcmp(err.identifier, ['bvar:forecast:simulate:' bad{ib,2}]), ...
            'call %d gave %s, expected %s', ib, err.identifier, bad{ib,2});
    end
end
end
