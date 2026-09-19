function test_var_sv
% bvar.models.var_sv must reproduce, draw for draw under one seed, the inline
% sampler ex06 used before the function existed (frozen below as mcmc_ex06), for
% both models and both orders, with 'phi_proposal' set to 'untruncated', the phi
% step of that sampler; and its embedded constants must equal the preset of the
% Chan, Koop and Yu (2024) package. Under the default truncated proposal every phi
% must move. With 'draws' it must run the same chain and return draws that average
% to its posterior means.
root = getappdata(0, 'bvar_repo_root');
pkg = fullfile(root, 'replications', 'chan_koop_yu2024_jbes_oisv');
od = cd(pkg); c = onCleanup(@() cd(od));
pr = preset();
clear c

assert(pr.ls_ridge == .01 && pr.sv_offset == .0001 && pr.phi_init_bnd == .99 ...
    && pr.oi.h_mean_in_sv == 0 && isequaln(pr.oi.kappa_init, [.1 .1 NaN 100]) ...
    && isequal(pr.cs.kappa_init, [.1 .1 1 100]), ...
    'var_sv: the package preset changed; update the constants in bvar.models.var_sv');

data = load(fullfile(pkg, 'legacy', pr.data_file));
p = pr.p; n0 = pr.n0;
nsim = 20; burnin = 10; seed = 20260915;

for rev = [false true]
    ord = pr.var_id1;
    if rev, ord = ord(end:-1:1); end
    Y0 = data(1:n0, ord);
    Y  = data(n0+1:end, ord);
    [~, X] = bvar.util.build_lags([Y0(end-p+1:end,:); Y], p);
    for model = {'OI', 'CS'}
        rng(seed, 'twister');
        ref = mcmc_ex06(model{1}, Y, X, Y0, p, nsim, burnin, pr);
        got = bvar.models.var_sv(Y0, Y, p, 'model', model{1}, 'nsim', nsim, ...
            'burnin', burnin, 'seed', seed, 'phi_proposal', 'untruncated');
        assert(isequal(got.Sig_mean, ref.Sig_mean) && isequal(got.kappa_mean, ref.kappa_mean), ...
            'var_sv(%s, reversed = %d): differs from the ex06 sampler', model{1}, rev);
    end
end

    % no seed: the current stream is used
rng(5, 'twister');
ref = mcmc_ex06('OI', Y, X, Y0, p, 5, 2, pr);
rng(5, 'twister');
got = bvar.models.var_sv(Y0, Y, p, 'nsim', 5, 'burnin', 2, 'phi_proposal', 'untruncated');
assert(isequal(got.Sig_mean, ref.Sig_mean), 'var_sv: without a seed it must use the current stream');

    % the default phi step, with the candidate from the truncated normal, moves
    % every phi within a short chain
r = bvar.models.var_sv(Y0, Y, p, 'nsim', 30, 'burnin', 5, 'seed', 4, 'draws', true);
assert(all(any(diff(r.draws.phi) ~= 0)), 'var_sv: under the default phi_proposal every phi must move');

    % the other outputs
[T, n] = size(Y); k = 1 + n*p;
got = bvar.models.var_sv(Y0, Y, p, 'model', 'CS', 'nsim', 5, 'burnin', 2, 'seed', 1);
assert(isequal(size(got.A_mean), [k n]) && isequal(size(got.h_mean), [T n]) ...
    && isequal(size(got.impact_mean), [n n]), 'var_sv: output sizes');
assert(all(diag(got.impact_mean) == 1) && all(all(triu(got.impact_mean, 1) == 0)), ...
    'var_sv: under ''CS'' the impact matrix must be unit lower triangular');
assert(all(isfinite(got.A_mean(:))) && all(isfinite(got.h_mean(:))), 'var_sv: non-finite output');

    % 'draws': the same chain, and draws that average to the means
for model = {'OI', 'CS'}
    a = bvar.models.var_sv(Y0, Y, p, 'model', model{1}, 'nsim', 6, 'burnin', 2, 'seed', 3);
    b = bvar.models.var_sv(Y0, Y, p, 'model', model{1}, 'nsim', 6, 'burnin', 2, 'seed', 3, ...
        'draws', true);
    assert(isequal(rmfield(b, 'draws'), a), 'var_sv(%s): ''draws'' changed the chain', model{1});
    D = b.draws;
    assert(isequal(size(D.kappa), [6 2]) && isequal(size(D.A), [6 k*n]) ...
        && isequal(size(D.impact), [6 n^2]) && isequal(size(D.phi), [6 n]) ...
        && isequal(size(D.sig2), [6 n]) && isfield(D, 'mu') == strcmp(model{1}, 'CS'), ...
        'var_sv(%s): draw sizes', model{1});
    assert(isequal(mean(D.kappa)', b.kappa_mean) ...
        && max(abs(mean(D.A) - b.A_mean(:)')) < 1e-12*max(abs(b.A_mean(:))) ...
        && max(abs(mean(D.impact) - b.impact_mean(:)')) < 1e-12*max(abs(b.impact_mean(:))), ...
        'var_sv(%s): the draws do not average to the posterior means', model{1});
end

    % input checks
Ybad = Y; Ybad(end, 1) = NaN;
expect_error(@() bvar.models.var_sv(Y0, Ybad, p, 'nsim', 1, 'burnin', 0), 'bvar:models:var_sv:badData');
expect_error(@() bvar.models.var_sv(Y0(1:3,:), Y, 2, 'nsim', 1, 'burnin', 0), 'bvar:models:var_sv:badData');
expect_error(@() bvar.models.var_sv(Y0, Y(:,1), p, 'nsim', 1, 'burnin', 0), 'bvar:models:var_sv:badData');
expect_error(@() bvar.models.var_sv(Y0, Y, p, 'model', 'XX'), 'bvar:models:var_sv:badModel');
expect_error(@() bvar.models.var_sv(Y0, Y, p, 'nsims', 5), 'bvar:models:var_sv:badOption');
expect_error(@() bvar.models.var_sv(Y0, Y, p, 'nsim', 0), 'bvar:models:var_sv:badOption');
expect_error(@() bvar.models.var_sv(Y0, Y, p, 'nsim', 1, 'burnin', 0, 'draws', 'yes'), ...
    'bvar:models:var_sv:badOption');
expect_error(@() bvar.models.var_sv(Y0, Y, p, 'nsim', 1, 'burnin', 0, 'phi_proposal', 'nw'), ...
    'bvar:models:var_sv:badOption');
end

function expect_error(f, id)
try
    f();
catch err
    assert(strcmp(err.identifier, id), 'expected %s, got %s: %s', id, err.identifier, err.message);
    return
end
error('expected error %s, none thrown', id);
end

function res = mcmc_ex06(model, Y, X, Y0, p, nsim, burnin, pr)
% The local sampler of examples/ex06_variable_ordering_sv.m as of 2026-09-18,
% frozen verbatim as the reference for bvar.models.var_sv.
[T, n] = size(Y);
k = 1 + n*p;
is_oi = strcmp(model, 'OI');

    % Minnesota second moments, built from AR(4) residual variances
sig2_ar = bvar.priors.resid_var_ar4(Y0, Y);
[C, idx_kappa1, idx_kappa2] = bvar.priors.minnesota_C(n, p, sig2_ar);

    % priors: the preset values of the package, at this n
Hyper.nuh = 3*ones(n,1);
Hyper.Sh = .05*(Hyper.nuh - 1);
Hyper.phi0 = .95*ones(n,1);
Hyper.Vphi = .05^2*ones(n,1);
Hyper.mu0 = zeros(n,1);
Hyper.Vmu = 100*ones(n,1);
Hyper.B0 = eye(n);
Hyper.VB0 = ones(n);
Hyper.beta0 = zeros(n^2*p + n, 1);
Hyper.Valp = ones(n*(n-1)/2, 1);
kappa = pr.oi.kappa_init;                    % [.1 .1 NaN 100]
if ~is_oi, kappa = pr.cs.kappa_init; end     % [.1 .1 1 100]

    % chain init
A = (X'*X + pr.ls_ridge*speye(k))\(X'*Y);
U = Y - X*A;
Sig_hat = U'*U/T;
h = repmat(log(diag(Sig_hat))', T, 1);
sig2 = 1./gamrnd(Hyper.nuh, 1./Hyper.Sh);
phi = min(Hyper.phi0 + sqrt(Hyper.Vphi).*randn(n,1), pr.phi_init_bnd);
z_psi1 = 1./gamrnd(.5, 1, n*p, 1);
z_psi2 = 1./gamrnd(.5, 1, (n-1)*n*p, 1);
z_kappa = 1./gamrnd(.5, 1, 2, 1);
Psi = ones(k*n, 1);
Psi(idx_kappa1) = 1./gamrnd(.5, z_psi1);
Psi(idx_kappa2) = 1./gamrnd(.5, z_psi2);
if is_oi
    B0 = diag(1./sqrt(diag(Sig_hat)));       % full matrix, updated row by row
else
    A_id = nonzeros(tril(reshape(1:n^2, n, n), -1)');
    Atri = eye(n); B = A'; XB = X*B';        % unit lower triangular impact matrix
    mu = zeros(n,1);
    for ii = 1:n, mu(ii) = mean(log(U(:,ii).^2)); end
end

Sig_sum = zeros(T, n, n);
store_kappa = zeros(nsim, 2);
for isim = 1:nsim + burnin
    [~, tmpdV] = bvar.priors.vtheta(idx_kappa1, idx_kappa2, kappa, C.*Psi, sig2);

    if is_oi
        B0 = bvar.structural.b0_row_sampler(Y - X*A, h, B0, Hyper.B0, Hyper.VB0);
        A = bvar.samplers.eq_var_oi(Y, X, B0, h, A, tmpdV);
        theta = A(:);
        E = (Y - X*A)*B0';                   % structural innovations
    else
        [B, XB] = bvar.samplers.eq_tri_cs(Y, X, XB, B, Atri, h, tmpdV, Hyper.beta0);
        theta = reshape(B', n^2*p + n, 1);
        E = Y - XB;
        Atri(A_id) = bvar.samplers.alp_tri_cs(E, h, Hyper.Valp);
        E = E*sparse(Atri');                 % structural innovations
    end

    for ii = 1:n
        ystar = log(E(:,ii).^2 + pr.sv_offset);
        if is_oi
            h(:,ii) = bvar.sv.ksc_ar1_mean(ystar, h(:,ii), pr.oi.h_mean_in_sv, phi(ii), sig2(ii));
        else
            h(:,ii) = bvar.sv.ksc_ar1_mean(ystar, h(:,ii), mu(ii), phi(ii), sig2(ii));
        end
    end
    if is_oi
        [phi, sig2] = bvar.sv.sv0_params(h, phi, Hyper);
    else
        [mu, phi, sig2] = bvar.sv.sv_params(h, mu, phi, Hyper);
    end

    [psi1, psi2, z_psi1, z_psi2, kappa, z_kappa] = bvar.samplers.horseshoe_kappa_psi( ...
        theta, idx_kappa1, idx_kappa2, C, kappa, z_psi1, z_psi2, z_kappa);
    Psi(idx_kappa1) = psi1;
    Psi(idx_kappa2) = psi2;

    if isim > burnin
        store_kappa(isim - burnin, :) = kappa(1:2);
        if is_oi
            Sig_sum = Sig_sum + bvar.structural.construct_Sigt(h, B0);
        else
            Sig_sum = Sig_sum + bvar.structural.construct_Sigt(h, Atri);
        end
    end
end

res.Sig_mean = Sig_sum/nsim;
res.kappa_mean = mean(store_kappa)';
end
