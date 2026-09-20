function test_forecast_predictive
% bvar.forecast.predictive against an independent companion-form computation:
% z_t = c~ + F*z_{t-1} + J'*e_t gives mean J*(sum_{j<h} F^j*c~ + F^h*z_T) and
% variance J*(sum_{j<h} F^j*Q*F^j')*J' with Q = J'*Sig*J, which shares no code
% with the moving-average recursion under test. Also checks a VAR(1) by hand,
% that each row of the draws is treated on its own, that tgt selects columns,
% and the argument checks.
rng(20260920, 'twister');

    % ---- VAR(1), two variables, one draw, worked out in the open ----
n = 2; p = 1; H = 3;
Phi = [.5 .1; -.2 .7];
c = [1; -.5];
Sig = [1 .3; .3 2];
A = reshape([c'; Phi'], 1, n*(n*p+1));
yT = [2; 1];
[mu, sd] = bvar.forecast.predictive(A, reshape(Sig, 1, n, n), yT, H);
m1 = c + Phi*yT;  m2 = c + Phi*m1;  m3 = c + Phi*m2;
V1 = Sig;  V2 = Sig + Phi*Sig*Phi';  V3 = V2 + Phi^2*Sig*(Phi^2)';
D = squeeze(mu) - [m1 m2 m3];
assert(max(abs(D(:))) < 1e-13, 'VAR(1) mean differs');
D = squeeze(sd) - sqrt([diag(V1) diag(V2) diag(V3)]);
assert(max(abs(D(:))) < 1e-13, 'VAR(1) standard deviation differs');

    % ---- random draws of a VAR(p) against the companion form ----
for cs = {[3 2 8], [2 1 5], [4 3 6]}
    [n, p, H] = deal(cs{1}(1), cs{1}(2), cs{1}(3));
    k = n*p + 1;  nsim = 5;
    A = zeros(nsim, n*k);  Sig = zeros(nsim, n, n);  ylag = randn(n, p);
    for d = 1:nsim
        Ad = [randn(1, n); .3*randn(n*p, n)/sqrt(p)];     % stationary enough for h up to 8
        S = randn(n);  S = S*S'/n + eye(n);
        A(d,:) = reshape(Ad, 1, n*k);
        Sig(d,:,:) = S;
    end
    [mu, sd] = bvar.forecast.predictive(A, Sig, ylag, H);
    for d = 1:nsim
        Ad = reshape(A(d,:), k, n);
        S = reshape(Sig(d,:,:), n, n);
        F = [Ad(2:end,:)'; eye(n*(p-1), n*p)];            % companion matrix
        ct = [Ad(1,:)'; zeros(n*(p-1), 1)];
        J = [eye(n), zeros(n, n*(p-1))];
        Q = J'*S*J;
        z = ylag(:);                                      % most recent lag first
        Fj = eye(n*p);  mz = z;  Vz = zeros(n*p);
        for h = 1:H
            mz = ct + F*mz;
            Vz = Vz + Fj*Q*Fj';
            Fj = F*Fj;
            assert(max(abs(mu(d,:,h)' - J*mz)) < 1e-11, ...
                'n=%d p=%d draw %d: mean differs at h=%d', n, p, d, h);
            assert(max(abs(sd(d,:,h)' - sqrt(diag(J*Vz*J')))) < 1e-11, ...
                'n=%d p=%d draw %d: standard deviation differs at h=%d', n, p, d, h);
        end
    end

        % rows are independent: one draw alone matches its row in the batch
    [mu1, sd1] = bvar.forecast.predictive(A(3,:), Sig(3,:,:), ylag, H);
    assert(isequal(mu1, mu(3,:,:)) && isequal(sd1, sd(3,:,:)), ...
        'n=%d p=%d: a single draw differs from its row of the batch', n, p);

        % tgt selects columns and nothing else
    tgt = [n 1];
    [mut, sdt] = bvar.forecast.predictive(A, Sig, ylag, H, tgt);
    assert(isequal(mut, mu(:,tgt,:)) && isequal(sdt, sd(:,tgt,:)), ...
        'n=%d p=%d: tgt does not select columns', n, p);
    assert(isequal(bvar.forecast.predictive(A, Sig, ylag, H, []), mu), ...
        'n=%d p=%d: empty tgt is not the default', n, p);
end

    % ---- argument checks ----
n = 3; p = 2; k = n*p + 1;
A = zeros(1, n*k);  S = reshape(eye(n), 1, n, n);  ylag = zeros(n, p);
bad = { @() bvar.forecast.predictive(A(1:end-1), S, ylag, 2), 'badDims'; ...
        @() bvar.forecast.predictive(A, S, ylag', 2),         'badLags'; ...
        @() bvar.forecast.predictive(A, S, ylag, 2, [1 n+1]), 'badTarget'};
for ib = 1:size(bad, 1)
    try
        bad{ib,1}();
        error('call %d should have errored (%s)', ib, bad{ib,2});
    catch err
        assert(strcmp(err.identifier, ['bvar:forecast:predictive:' bad{ib,2}]), ...
            'call %d gave %s, expected %s', ib, err.identifier, bad{ib,2});
    end
end
end
