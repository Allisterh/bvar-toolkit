function test_kron_ml_densities
% deterministic equivalence of the step-8 density/likelihood utilities with
% their chan2020_jbes_kronecker legacy copies:
%   bvar.ml.lniwpdf      = legacy lniwpdf.m
%   bvar.ml.linvgammpdf  = legacy linvgammpdf.m
%   bvar.ml.llike_ma     = legacy llike_MA.m (root)
%   bvar.ml.llike_csv_ma = legacy llike_CSV_MA.m (package ROOT copy)
% linvgammpdf and llike_csv_ma bitwise. lniwpdf and llike_ma take the lower
% Cholesky factor where the legacy copies take the upper one; for a dense
% matrix the two factors can differ in the last bits, so these two are
% checked bitwise against legacy copies carrying the same three
% substitutions as in test_kron_equivalence, and against the unmodified
% copies to within 1e-12 relative, at n = 4 and at n = 20.
% Plus the never-merge direction check: the realtime_forecasts copy of
% llike_CSV_MA omits the -n/2*sum(h) term, so the two must DIFFER by
% n/2*sum(h) (up to one-rounding tolerance - the term is folded into the
% root copy's constant before the common subtraction).
root = getappdata(0, 'bvar_repo_root');
leg = fullfile(root, 'replications', 'chan2020_jbes_kronecker', 'legacy');
legrt = fullfile(leg, 'realtime_forecasts');

% --- synthetic evaluation points (deterministic given the seed) ---
rng(8, 'twister');
T = 25; n = 4; k = 9;
A = randn(k, n); A0 = randn(k, n)/5;
Q = randn(n, n+3); Sig = Q*Q'/(n+3); Sig = (Sig+Sig')/2;
Q = randn(n, n+5); S0 = Q*Q'/(n+5); S0 = (S0+S0')/2;
VA0 = 1./gamrnd(3, 1, k, 1);
iVA0 = sparse(1:k, 1:k, 1./VA0);
nu0 = n+3;
U = randn(T, n)*chol(Sig, 'lower')';
h = .3*randn(T, 1);
psi = .15;

% 20 variables, with a dense iVA0 of the form of the posterior precision KA
T2 = 60; n2 = 20; k2 = 1 + 4*n2;
A2 = randn(k2, n2); A02 = randn(k2, n2)/5;
Q = randn(n2, n2+3); Sig2 = Q*Q'/(n2+3); Sig2 = (Sig2+Sig2')/2;
Q = randn(n2, n2+5); S02 = Q*Q'/(n2+5); S02 = (S02+S02')/2;
X2 = randn(T2, k2); iVA02 = diag(gamrnd(3, 1, k2, 1)) + X2'*X2;
nu02 = n2+3;
U2 = randn(T2, n2)*chol(Sig2, 'lower')';

% --- core values ---
c_lniw = bvar.ml.lniwpdf(A, Sig, A0, iVA0, nu0, S0);
c_lniw2 = bvar.ml.lniwpdf(A2, Sig2, A02, iVA02, nu02, S02);
yv = [.5 1.2 3]; av = [2 3.5 4]; bv = [.5 .8 2];
c_ligp = bvar.ml.linvgammpdf(yv, av, bv);
c_lma = bvar.ml.llike_ma(psi, U, Sig);
c_lma2 = bvar.ml.llike_ma(psi, U2, Sig2);
c_lcsvma = bvar.ml.llike_csv_ma(psi, U, Sig, h);

% --- unmodified legacy root copies ---
addpath(leg); c1 = onCleanup(@() rmpath(leg));
assert(strncmpi(which('lniwpdf'), leg, numel(leg)), 'lniwpdf must resolve from legacy');
near = @(a, b) abs(a - b) <= 1e-12*abs(b);
assert(near(c_lniw, lniwpdf(A, Sig, A0, iVA0, nu0, S0)) ...
    && near(c_lniw2, lniwpdf(A2, Sig2, A02, iVA02, nu02, S02)), ...
    'bvar.ml.lniwpdf is not within 1e-12 of legacy lniwpdf');
assert(isequal(linvgammpdf(yv, av, bv), c_ligp), ...
    'bvar.ml.linvgammpdf differs from legacy linvgammpdf');
assert(near(c_lma, llike_MA(psi, U, Sig)) && near(c_lma2, llike_MA(psi, U2, Sig2)), ...
    'bvar.ml.llike_ma is not within 1e-12 of legacy llike_MA');
l_root = llike_CSV_MA(psi, U, Sig, h);
assert(isequal(l_root, c_lcsvma), ...
    'bvar.ml.llike_csv_ma differs from the legacy ROOT llike_CSV_MA');
clear c1

% --- lniwpdf.m and llike_MA.m with the lower-Cholesky substitution, bitwise ---
tmp = tempname; mkdir(tmp);
c3 = onCleanup(@() cleanup_tmp(tmp));
lower_subs = {'llike_MA.m', 'CSig = chol(Sig)'';', 'CSig = chol(Sig,''lower'');'; ...
    'lniwpdf.m', 'diag(chol(iVA0))', 'diag(chol(iVA0,''lower''))'; ...
    'lniwpdf.m', 'diag(chol(S0))', 'diag(chol(S0,''lower''))'};
for kf = unique(lower_subs(:,1))'
    txt = fileread(fullfile(leg, kf{1}));
    for ks = find(strcmp(lower_subs(:,1), kf{1}))'
        assert(numel(strfind(txt, lower_subs{ks,2})) == 1, ...
            'expected exactly one %s in %s', lower_subs{ks,2}, kf{1});
        txt = strrep(txt, lower_subs{ks,2}, lower_subs{ks,3});
    end
    fid = fopen(fullfile(tmp, kf{1}), 'w');
    fwrite(fid, txt);
    fclose(fid);
end
addpath(tmp);
assert(strncmpi(which('lniwpdf'), tmp, numel(tmp)), ...
    'lniwpdf must resolve from the substituted copy');
assert(isequal(lniwpdf(A, Sig, A0, iVA0, nu0, S0), c_lniw) ...
    && isequal(lniwpdf(A2, Sig2, A02, iVA02, nu02, S02), c_lniw2), ...
    'bvar.ml.lniwpdf differs from the substituted legacy lniwpdf');
assert(isequal(llike_MA(psi, U, Sig), c_lma) && isequal(llike_MA(psi, U2, Sig2), c_lma2), ...
    'bvar.ml.llike_ma differs from the substituted legacy llike_MA');
clear c3

% --- realtime copy: must differ by exactly the -n/2*sum(h) term ---
addpath(legrt); c2 = onCleanup(@() rmpath(legrt));
assert(strncmpi(which('llike_CSV_MA'), legrt, numel(legrt)), ...
    'llike_CSV_MA must now resolve from realtime_forecasts');
l_rt = llike_CSV_MA(psi, U, Sig, h);
assert(~isequal(l_rt, l_root), ...
    'root and realtime llike_CSV_MA should differ (never-merge)');
gap = (l_rt - l_root) - n/2*sum(h);
assert(abs(gap) < 1e-9*max(1, abs(l_root)), ...
    'root vs realtime llike_CSV_MA difference is not the n/2*sum(h) term (gap %.3g)', gap);
clear c2
end

function cleanup_tmp(tmp)
% path entry first, then the folder itself
if any(strcmpi(strsplit(path, pathsep), tmp))
    rmpath(tmp);
end
if exist(tmp, 'dir')
    rmdir(tmp, 's');
end
end
