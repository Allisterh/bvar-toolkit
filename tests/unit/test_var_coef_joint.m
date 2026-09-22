function test_var_coef_joint
% bvar.samplers.var_coef_joint must draw vec(A) from the Gaussian with precision
% iVb + kron(Sig^{-1}, X'*W*X) and mean that precision's inverse times vec(X'*W*Y*Sig^{-1}),
% which is computed here with dense algebra from the likelihood written period by period.
rng(20260921, 'twister');
T = 40; n = 2; k = 3;
X = [ones(T,1) randn(T,2)];
Y = randn(T,n);
Sig = [1 .4; .4 .7];
w = exp(-1.5*sin((1:T)'/4));                      % strongly varying weights
V = [4 1 .5; 2 .3 .2]';                           % k x n prior variances
iVb = diag(1./V(:));

% dense: the sum over periods of kron(Sig^{-1}, x_t*x_t')*w_t and the matching linear term
iS = inv(Sig);
P = iVb;  c = zeros(k*n,1);
for t = 1:T
    P = P + w(t)*kron(iS, X(t,:)'*X(t,:));
    c = c + w(t)*kron(iS, X(t,:)')*Y(t,:)';
end
mu = P\c;  Vp = inv(P);

[~, Ahat] = bvar.samplers.var_coef_joint(Y, X, Sig, w, sparse(iVb));
assert(norm(Ahat(:) - mu, inf) < 1e-10, 'var_coef_joint: the posterior mean differs from dense algebra');
nd = 40000;
D = zeros(nd, k*n);
for i = 1:nd
    A = bvar.samplers.var_coef_joint(Y, X, Sig, w, sparse(iVb));
    D(i,:) = A(:)';
end
zs = (mean(D)' - mu)./sqrt(diag(Vp)/nd);
assert(max(abs(zs)) < 5, 'var_coef_joint: the draws are off the dense mean by %.1f Monte Carlo sd', max(abs(zs)));
C = cov(D);
assert(max(abs(C(:) - Vp(:))) < .03*max(abs(Vp(:))), 'var_coef_joint: the covariance of the draws differs');

% the weights enter as precisions: w = 1 everywhere is the homoskedastic posterior
[~, A1] = bvar.samplers.var_coef_joint(Y, X, Sig, ones(T,1), sparse(iVb));
P1 = iVb + kron(iS, X'*X);
assert(norm(A1(:) - P1\reshape(X'*Y*iS, [], 1), inf) < 1e-10, ...
    'var_coef_joint: unit weights must give the homoskedastic posterior mean');
fprintf('test_var_coef_joint passed\n');
end
