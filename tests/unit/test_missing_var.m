function test_missing_var
% bvar.samplers.missing_var must give the conditional distribution of the missing values of
% a VAR(p) given the observed values and the restrictions M*y = z, hard or soft, which is
% computed here a second way, by conditioning the joint normal of the whole stacked path
% with dense algebra.
rng(20260921, 'twister');
n = 3; p = 2; T = 36;
A = [0.2 -0.1 0.3; 0.5 0.1 0; 0.1 0.4 0.1; 0 0.1 0.3; 0.1 0 0; 0 -0.1 0.1; 0 0 0.1];
CS = [1 0 0; .4 .8 0; .2 .3 .6];  Sig = CS*CS';
h = 0.5*sin((1:T-p)'/5);                           % a moving volatility path

% one generated path, started at zero
Yt = zeros(T, n);
for t = p+1:T
    x = [1, reshape(Yt(t-1:-1:t-p,:)', 1, n*p)];
    Yt(t,:) = x*A + exp(h(t-p)/2)*randn(1,n)*CS';
end

% the third series is seen only through Mariano-Murasawa quarterly values, the first has a
% hole, and the second a ragged edge
X = Yt;
X(:,3) = NaN;
qend = (6:3:T)';
y3 = Yt(:,3);
X(qend,3) = y3(qend - (0:4))*[1 2 3 2 1]'/3;
[M, z, Y] = bvar.util.mm_constraint(X, [false false true]);
Y(14,1) = NaN;  Y(T,2) = NaN;
% one more restriction, on two missing values and an observed one: y(14,1) + y(14,3) +
% y(15,1) = 0.7
M = [M; sparse([1 1 1], [13*n+1, 13*n+3, 14*n+1], 1, 1, T*n)];
z = [z; 0.7];
V0 = [4; 9; 25];  m0 = [0.1; -0.2; 0.3];

[ym, ymhat, S] = bvar.samplers.missing_var(Y, A, Sig, h, p, 'M', M, 'z', z, ...
    'V0', V0, 'm0', m0, 'ndraws', 40000);

% the joint normal of y = vec(Y'): each period after the first p has its VAR equation, and
% each value in the first p periods its N(m0, V0) prior, which drops out for observed ones
Te = T - p;
H = kron(sparse(1:Te, p+1:T, 1, Te, T), speye(n));
for l = 1:p
    H = H - kron(sparse(1:Te, (p+1:T)-l, 1, Te, T), sparse(A(1+(l-1)*n+1:1+l*n,:)'));
end
iOm = kron(diag(exp(-h)), inv(Sig));
P0 = kron(diag([ones(p,1); zeros(Te,1)]), diag(1./V0));
Q = full(H'*iOm*H) + P0;
b = full(H'*iOm*kron(ones(Te,1), A(1,:)')) + P0*kron(ones(T,1), m0);
Vy = inv(Q);  Vy = (Vy + Vy')/2;
mu = Vy*b;

% condition on the observed values and on M*y = z at once
L = [S.So'; M];
v = [S.yo; z];
G = Vy*L'/(L*Vy*L');
muc = mu + G*(v - L*mu);
Vc = Vy - G*L*Vy;
mdense = S.Sm'*muc;
Vdense = S.Sm'*Vc*S.Sm;  Vdense = (Vdense + Vdense')/2;

assert(norm(ymhat - mdense, inf) < 1e-8, 'missing_var: the conditional mean differs from dense conditioning');
sd = sqrt(diag(Vdense));
zs = (mean(ym, 2) - mdense)./(sd/sqrt(size(ym,2)));
assert(max(abs(zs)) < 5, 'missing_var: the mean of the draws is %.1f Monte Carlo sd from the dense mean', max(abs(zs)));
Cdraw = cov(ym');
assert(max(abs(Cdraw(:) - Vdense(:))) < 0.04*max(abs(Vdense(:))), ...
    'missing_var: the covariance of the draws differs from dense conditioning');
ycomp = S.So*S.yo + S.Sm*ym;
assert(max(max(abs(M*ycomp - z))) < 1e-9, 'missing_var: a draw breaks the restrictions');

% soft restrictions, z = M*y + e with e ~ N(0, diag(o)): condition on yo and z, whose joint
% covariance now has diag(o) added to the block of z
o = 0.05 + 0.1*rand(numel(z), 1);
[ys, yshat] = bvar.samplers.missing_var(Y, A, Sig, h, p, 'M', M, 'z', z, 'O', o, ...
    'V0', V0, 'm0', m0, 'ndraws', 40000);
No = numel(S.yo);
Cv = L*Vy*L' + blkdiag(zeros(No), diag(o));
Gs = Vy*L'/Cv;
ms = S.Sm'*(mu + Gs*(v - L*mu));
Vs = S.Sm'*(Vy - Gs*L*Vy)*S.Sm;  Vs = (Vs + Vs')/2;
assert(norm(yshat - ms, inf) < 1e-8, 'missing_var: the soft conditional mean differs from dense conditioning');
zs = (mean(ys, 2) - ms)./(sqrt(diag(Vs))/sqrt(size(ys,2)));
assert(max(abs(zs)) < 5, 'missing_var: soft draws are %.1f Monte Carlo sd from the dense mean', max(abs(zs)));
Cs = cov(ys');
assert(max(abs(Cs(:) - Vs(:))) < 0.04*max(abs(Vs(:))), ...
    'missing_var: the covariance of the soft draws differs from dense conditioning');
% tiny variances give the hard restrictions back
[~, ytiny] = bvar.samplers.missing_var(Y, A, Sig, h, p, 'M', M, 'z', z, 'O', 1e-10, 'V0', V0, 'm0', m0);
assert(norm(ytiny - ymhat, inf) < 1e-4, 'missing_var: soft restrictions with tiny variances must approach the hard ones');

% no restrictions: the unrestricted conditional, the same way
[~, ymhat0] = bvar.samplers.missing_var(Y, A, Sig, h, p, 'V0', V0, 'm0', m0);
L0 = S.So';
G0 = Vy*L0'/(L0*Vy*L0');
assert(norm(ymhat0 - S.Sm'*(mu + G0*(S.yo - L0*mu)), inf) < 1e-8, ...
    'missing_var: the unrestricted conditional mean differs from dense conditioning');

% nothing missing: empty output, and no random numbers used
s = rng;
[ym1, ymhat1] = bvar.samplers.missing_var(Yt, A, Sig, h, p);
assert(isempty(ym1) && isempty(ymhat1) && isequal(rng, s), ...
    'missing_var: a complete panel must return empty draws without using the stream');

% the argument checks
bad = { @() bvar.samplers.missing_var(Y, A(1:end-1,:), Sig, h, p), 'badSize'; ...
        @() bvar.samplers.missing_var(Y, A, Sig, h(1:end-1), p), 'badSize'; ...
        @() bvar.samplers.missing_var(Y, A, Sig, h, p, 'M', M), 'badRestriction'; ...
        @() bvar.samplers.missing_var(Y, A, Sig, h, p, 'M', [M; sparse(1,1,1,1,T*n)], 'z', [z; 0]), 'badRestriction'; ...
        @() bvar.samplers.missing_var(Y, A, Sig, h, p, 'O', 1), 'badRestriction'; ...
        @() bvar.samplers.missing_var(Y, A, Sig, h, p, 'M', M, 'z', z, 'O', [1 2]), 'badRestriction'; ...
        @() bvar.samplers.missing_var(Y, A, Sig, h, p, 'M', M, 'z', z, 'O', 0), 'badRestriction'; ...
        @() bvar.samplers.missing_var(Y, A, Sig, h, p, 'zzz', 1), 'badOption'};
for ib = 1:size(bad, 1)
    try
        bad{ib,1}();
        error('the call with a %s input should have errored', bad{ib,2});
    catch err
        assert(strcmp(err.identifier, ['bvar:samplers:missing_var:' bad{ib,2}]), ...
            'wrong identifier: %s', err.identifier);
    end
end
fprintf('test_missing_var passed\n');
end
