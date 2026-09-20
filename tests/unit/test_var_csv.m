function test_var_csv
% bvar.models.var_csv must reproduce, draw for draw under one seed, the inline
% sampler ex05 used before the function existed, frozen below as csv_chain_ex05,
% and must refuse the inputs its header rules out.
rng(20260920, 'twister');
n = 3; p = 2; T = 60;
Phi = [.5 .1 0; .1 .4 .1; 0 .1 .3];
CS = [1 0 0; .4 1 0; .2 .3 1]/2;
Yall = zeros(T+30, n);
for t = 2:T+30
    Yall(t,:) = Yall(t-1,:)*Phi' + (CS*randn(n,1))';
end
Y0 = Yall(27:30, :);  Y = Yall(31:end, :);

%% ---- 1. the default path, bitwise against the frozen ex05 sampler ----
nsim = 15; burnin = 5; seed = 4242;
ref = csv_chain_ex05(Y0, Y, p, nsim, burnin, 3, seed);
got = bvar.models.var_csv(Y0, Y, p, 'nsim', nsim, 'burnin', burnin, ...
    'seed', seed, 'c_reject', 3, 'draws', true);
assert(isequal(got.draws.h, ref.store_h), 'ncp: the h draws differ');
assert(isequal(got.draws.A, ref.store_A), 'ncp: the coefficient draws differ');
assert(isequal(got.draws.phi, ref.store_phi), 'ncp: the phi draws differ');
assert(isequal(got.draws.sigh2, ref.store_sigh2), 'ncp: the sigh2 draws differ');
assert(isequal(got.Sig_mean, ref.Sig_sum/nsim), 'ncp: the posterior mean of Sig differs');
assert(isequal(got.h_mean, mean(ref.store_h, 1)'), 'ncp: the posterior mean of h differs');
assert(abs(got.accept_rate - ref.accept_rate) < 1e-15, 'ncp: the acceptance rate differs');

%% ---- 2. the argument checks ----
bad = { @() bvar.models.var_csv(Y0, Y, p, 'nsim', 0), 'a nonpositive nsim'; ...
        @() bvar.models.var_csv(Y0, Y, p, 'zzz', 1), 'an unknown option'; ...
        @() bvar.models.var_csv(Y0, Y, p, 'nsim'), 'an option with no value'};
for ib = 1:size(bad, 1)
    try
        bad{ib,1}();
        error('%s should have errored', bad{ib,2});
    catch err
        assert(strcmp(err.identifier, 'bvar:models:var_csv:badOption'), ...
            'wrong identifier for %s: %s', bad{ib,2}, err.identifier);
    end
end
bad = { @() bvar.models.var_csv(Y0, Y(:,1), p), 'a single column'; ...
        @() bvar.models.var_csv(Y0(1:2,:), Y, p), 'too few initial conditions'; ...
        @() bvar.models.var_csv(Y0, [Y; nan(1,n)], p), 'a missing value'};
for ib = 1:size(bad, 1)
    try
        bad{ib,1}();
        error('%s should have errored', bad{ib,2});
    catch err
        assert(strcmp(err.identifier, 'bvar:models:var_csv:badData'), ...
            'wrong identifier for %s: %s', bad{ib,2}, err.identifier);
    end
end
end

% -------------------------------------------------------------------------
function out = csv_chain_ex05(Y0, Y, p, nsim, burnin, c_reject, seed)
% FROZEN copy of the local csv_chain of examples/ex05_var_csv.m, as it stood
% before bvar.models.var_csv existed. Do not edit to follow the library.
rng(seed, 'twister');
[T, n] = size(Y);
k = 1 + n*p;
[~, X] = bvar.util.build_lags([Y0(end-p+1:end,:); Y], p);
[A0, VA0, nu0, S0] = bvar.priors.niw(p, [.2^2 100], Y0, Y, 'mlvarsv_ncp');
Hyper = struct('nuh', 3, 'Sh', .2, 'phi0', .98, 'Vphi', .05^2);
phi = Hyper.phi0;
sigh2 = 1/gamrnd(Hyper.nuh, 1/Hyper.Sh);
h = zeros(T,1);
iVA0 = sparse(1:k, 1:k, 1./VA0);
VA0iA0 = sparse(1:k, 1:k, VA0)\A0;
store_h = zeros(nsim, T);
store_phi = zeros(nsim, 1);
store_sigh2 = zeros(nsim, 1);
store_A = zeros(nsim, k*n);
Sig_sum = zeros(n);
n_accept = 0; n_mh = 0;
for isim = 1:nsim + burnin
    iOh = sparse(1:T, 1:T, exp(-h));
    XiOh = X'*iOh;
    KA = iVA0 + XiOh*X;
    CKA = chol(KA, 'lower');
    Ahat = (CKA')\(CKA\(VA0iA0 + XiOh*Y));
    Shat = S0 + A0'*iVA0*A0 + Y'*iOh*Y - Ahat'*KA*Ahat;
    Shat = (Shat + Shat')/2;
    Sig = iwishrnd(Shat, nu0 + T);
    CSig = chol(Sig, 'lower');
    A = Ahat + (CKA'\randn(k,n))*CSig';
    U = Y - X*A;
    tmp = U/CSig';
    s2 = sum(tmp.^2, 2);
    if isim <= 20
        h = bvar.sv.csv_armh(s2, phi, sigh2, h, n, true, [], 'c_reject', c_reject);
    else
        [h, is_accept] = bvar.sv.csv_armh(s2, phi, sigh2, h, n, false, [], ...
            'c_reject', c_reject);
        n_accept = n_accept + is_accept;
        n_mh = n_mh + 1;
    end
    [phi, sigh2] = bvar.sv.sv0_params(h, phi, Hyper);
    if isim > burnin
        i = isim - burnin;
        store_h(i,:) = h';
        store_phi(i) = phi;
        store_sigh2(i) = sigh2;
        store_A(i,:) = A(:)';
        Sig_sum = Sig_sum + Sig;
    end
end
out = struct('store_h', store_h, 'store_phi', store_phi, 'store_sigh2', store_sigh2, ...
    'store_A', store_A, 'Sig_sum', Sig_sum, 'accept_rate', n_accept/max(n_mh,1));
end
