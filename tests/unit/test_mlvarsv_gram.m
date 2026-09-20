function test_mlvarsv_gram
% the 'gram' option of the three ml_varsv marginal-likelihood routines. Under
% 'blocks' the weighted Gram matrix in the precision of the VAR coefficients is
% formed from its k x k blocks, where the default 'full' multiplies out the full
% design matrix; the two draw the same random numbers and agree to rounding. The
% comparison runs the whole pipeline at n = 5 on the package's data.
root = getappdata(0, 'bvar_repo_root');
repdir = fullfile(root, 'replications', 'chan2023_joe_mlvarsv');
addpath(repdir);
cp = onCleanup(@() rmpath(repdir));

    % bvar.util.kron_gram against the products it replaces
rng(20260919, 'twister');
T = 20; k = 7; n = 4;
X = randn(T, k);  B0 = eye(n) + tril(randn(n), -1);
W = exp(randn(T, n));  U = randn(T, n);
d = bvar.util.vec(1./sqrt(W));
Xt = kron(B0, X)./d;  yt = bvar.util.vec(U)./d;
[XWX, XWy, yWy] = bvar.util.kron_gram(X, B0, W, U);
rel = @(a, b) max(abs(a(:) - b(:)))/max(abs(b(:)));
assert(rel(XWX, Xt'*Xt) < 1e-12 && rel(XWy, Xt'*yt) < 1e-12 && rel(yWy, yt'*yt) < 1e-12, ...
    'kron_gram differs from the products it replaces');
assert(issymmetric(round(XWX, 10)), 'kron_gram is not symmetric');

    % the three routines, through run_ml
nsim = 60; burnin = 10; M = 100; seed = 20260919;
varid = [1,22,59,120,133];              % n = 5, the first five of the n = 7 selection
for m = [3 4 5]
    [~, a] = evalc('run_ml(m, [], [], nsim, burnin, seed, varid, ''M'', M)');
    sa = rng;
    [~, b] = evalc('run_ml(m, [], [], nsim, burnin, seed, varid, ''M'', M, ''gram'', ''blocks'')');
    sb = rng;
    assert(strcmp(a.ml_gram, 'full') && strcmp(b.ml_gram, 'blocks'), 'the option is not recorded');
    assert(isequal(sa, sb), 'model %d: the two settings draw different random numbers', m);
    fitted = setdiff(fieldnames(a.ml), {'store_w', 'bigml'});   % the importance densities
    for f = fitted'
        assert(isequal(a.ml.(f{1}), b.ml.(f{1})), 'model %d: %s differs', m, f{1});
    end
    dw = max(abs(a.ml.store_w - b.ml.store_w));
    assert(dw < 1e-6, 'model %d: log weights differ by %.2g', m, dw);
    assert(abs(a.lml - b.lml) < 1e-6 && abs(a.lmlstd - b.lmlstd) < 1e-6, ...
        'model %d: the estimate differs by %.2g', m, abs(a.lml - b.lml));
end

    % invalid values
for f = {@() run_ml(3, [], [], nsim, burnin, seed, varid, 'M', M, 'gram', 'sparse'), ...
         @() run_ml(4, [], [], nsim, burnin, seed, varid, 'M', M, 'gram', 'sparse'), ...
         @() run_ml(5, [], [], nsim, burnin, seed, varid, 'M', M, 'gram', 'sparse')}
    try
        evalc('f{1}()');
        error('expected an error for an unknown gram value');
    catch err
        assert(contains(err.identifier, 'badGram'), 'expected badGram, got %s', err.identifier);
    end
end
end
