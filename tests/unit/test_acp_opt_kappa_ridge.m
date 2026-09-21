function test_acp_opt_kappa_ridge
% The two legacy copies of get_OptKappa differ only in the marginal likelihood
% they call: the ACP package's ml_VAR_ACP forms the posterior precision as
% iVi + Xi'*Xi, the SVAR-sign package's adds 1e-6*speye(ki). bvar.ml.acp covers
% both through 'ridge', and this pins the optimizer that calls it: the default
% reproduces the first copy, 'ridge', 1e-6 the second, and the two disagree, so
% the option cannot become a no-op.
rng(7, 'twister');
n = 4; p = 2; T = 80; n0 = 8;
A1 = .5*eye(n) + .05*randn(n);  A2 = .2*eye(n);
burn = 20;
D = zeros(T + n0 + burn, n);
for t = p+1:size(D,1)
    D(t,:) = D(t-1,:)*A1' + D(t-2,:)*A2' + .5*randn(1,n);
end
D = D(burn+1:end,:);
Y0 = D(1:n0,:);  Y = D(n0+1:end,:);
tmpY = [Y0(end-p+1:end,:); Y];
Z = zeros(T, n*p);
for ii = 1:p, Z(:,(ii-1)*n+1:ii*n) = tmpY(p-ii+1:end-ii,:); end
Z = [ones(T,1) Z];
k0 = [.04 .0016];  idx_ns = [1 3];

[m_def, k_def] = bvar.priors.acp_opt_kappa(Y0, Y, Z, p, k0, 'redu', idx_ns);
[m_rdg, k_rdg] = bvar.priors.acp_opt_kappa(Y0, Y, Z, p, k0, 'redu', idx_ns, 'ridge', 1e-6);

root = getappdata(0, 'bvar_repo_root');
acp = fullfile(root, 'replications', 'chan2022_qe_acp', 'legacy', 'utility');
svs = fullfile(root, 'replications', 'chan_matthes_yu2026_qe_svarsign', 'legacy', 'utility');
old = path;  c = onCleanup(@() path(old));   %#ok<NASGU> restores whichever copy is on it

addpath(acp);
[m1, k1] = get_OptKappa(Y0, Y, Z, p, k0, 'redu', idx_ns);
path(old);
assert(isequal(m1, m_def) && isequal(k1, k_def), ...
    'the default differs from the ACP package''s get_OptKappa');

addpath(svs);
[m2, k2] = get_OptKappa(Y0, Y, Z, p, k0, 'redu', idx_ns);
path(old);
assert(isequal(m2, m_rdg) && isequal(k2, k_rdg), ...
    '''ridge'', 1e-6 differs from the SVAR-sign package''s get_OptKappa');

assert(~isequal(m1, m2), 'the ridge left the objective unchanged on this data');
end
