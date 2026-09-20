function test_forecast_simulate
% bvar.forecast.simulate must equal bvar.forecast.predictive at every horizon
% when the covariance is constant, must reduce to that case when the volatility
% is switched off, must scale the covariance by exp(h) when the log-volatility is
% held at a constant, and must check its arguments.
rng(20260921, 'twister');
n = 3; p = 2;
A = [0.2 -0.1 0.05; 0.5 0.1 0; 0.1 0.4 0.1; 0 0.2 0.5; 0.1 0 0; 0 0.1 0; 0 0 0.1];
CS = [1 0 0; .4 1 0; .2 .3 1]/2;
Sig = CS*CS';
ylag = [0.3 -0.2; 0.1 0.4; -0.5 0.2];
H = 4;
yobs = [0.2 0.1 -0.3; 0.4 -0.1 0.2; -0.1 0.3 0.1; 0.2 0.2 -0.2];
cfg = struct('ylag', ylag, 'H', H, 'yobs', yobs);

%% ---- 1. the constant-covariance case, against the closed form ----
[yh, ld, lj, sd] = bvar.forecast.simulate('gauss', struct('A', A, 'Sig', Sig), cfg);
[mu, sdx] = bvar.forecast.predictive(A(:)', reshape(Sig, 1, n, n), ylag, H);
mu = reshape(mu, n, H)';  sdx = reshape(sdx, n, H)';
assert(max(abs(yh(:) - mu(:))) < 1e-14, ...
    'gauss: the conditional means must equal those of bvar.forecast.predictive');
assert(max(abs(sd(:) - sdx(:))) < 1e-13, ...
    'gauss: the standard deviations must equal those of bvar.forecast.predictive');
    % and the densities follow from those two
want = -.5*log(2*pi*sdx.^2) - .5*((yobs - mu)./sdx).^2;
assert(max(abs(ld(:) - want(:))) < 1e-12, 'gauss: the log densities are wrong');
assert(all(isfinite(lj)), 'gauss: the joint log densities must be finite');

%% ---- 2. the two volatility models with their innovations switched off ----
d = struct('A', A, 'Sig', Sig, 'h_T', 0, 'phi', 0, 'sigh2', 0);
[~, ld2, lj2, sd2] = bvar.forecast.simulate('csv', d, cfg);
assert(max(abs(ld2(:) - ld(:))) < 1e-12 && max(abs(sd2(:) - sd(:))) < 1e-13 ...
    && max(abs(lj2 - lj)) < 1e-12, ...
    'csv with a fixed volatility must match the constant-covariance case');

B0 = inv(CS);                                   % so B0\diag(1)/B0' = CS*CS' = Sig
d = struct('A', A, 'impact', B0, 'h_T', zeros(1,n), 'phi', zeros(1,n), 'sig2', zeros(1,n));
[~, ld3, ~, sd3] = bvar.forecast.simulate('oisv', d, cfg);
assert(max(abs(ld3(:) - ld(:))) < 1e-10 && max(abs(sd3(:) - sd(:))) < 1e-11, ...
    'oisv with a fixed volatility must match the constant-covariance case');

%% ---- 3. a constant log-volatility scales the covariance by its exponential ----
c = 0.7;
d = struct('A', A, 'Sig', Sig, 'h_T', c, 'phi', 1, 'sigh2', 0);
[~, ~, ~, sd4] = bvar.forecast.simulate('csv', d, cfg);
assert(max(abs(sd4(:) - exp(c/2)*sd(:))) < 1e-12, ...
    'csv: a constant log-volatility must scale every standard deviation by exp(h/2)');

%% ---- 4. a volatility path enters the variance horizon by horizon ----
    % with phi = 0 the path is iid, so run many draws and check that the average
    % variance at h = 1 matches E[exp(h)]*Sig = exp(sigh2/2)*Sig
sigh2 = 0.5;
d = struct('A', A, 'Sig', Sig, 'h_T', 0, 'phi', 0, 'sigh2', sigh2);
R = 20000;  acc = zeros(1, n);
for r = 1:R
    [~, ~, ~, s1] = bvar.forecast.simulate('csv', d, cfg);
    acc = acc + s1(1,:).^2;
end
want1 = exp(sigh2/2)*diag(Sig)';
assert(max(abs(acc/R - want1)./want1) < .05, ...
    'csv: the average variance at the first horizon is off its expectation');

%% ---- 5. argument checks ----
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
