%% ex07 - A VAR with factor stochastic volatility
%
% BOOK: Chapter 14, Section 14.4, in Bayesian Macroeconometrics: Methods and
% Applications (Chapman & Hall/CRC, forthcoming); the static factor model it
% builds on is Chapter 11.
%
% THE MODEL. The errors of the VAR load on r latent factors, and both the
% factors and the idiosyncratic errors carry their own stochastic volatility:
%
%       y_t = A'x_t + L f_t + u_t,   u_t ~ N(0, D_t),  f_t ~ N(0, G_t),
%       D_t = diag(exp(h_{1t}), ..., exp(h_{nt})),
%       G_t = diag(exp(h_{n+1,t}), ..., exp(h_{n+r,t})),
%
% so Sigma_t = L G_t L' + D_t. Here n = 8 and r = 2: the covariance moves with
% 10 log-volatility paths and 13 free loadings, against 36 free elements in an
% unrestricted covariance at every date.
%
% NORMALIZATION. L has ones on its diagonal and zeros above it, so the first r
% variables fix the scale of the factors and the factor log-volatilities have
% free means. Section 14.4 of the book normalizes the other way round, leaving L
% unrestricted and giving the factor log-volatilities zero means. Sigma_t is the
% same either way; what differs is which quantities are separately identified.
%
% The data are simulated from the model, so the factors, the loadings and the
% covariance paths all have a known truth. The sampler is the one of
% replications/chan2023_joe_mlvarsv/legacy/VAR_FSV.m, with the shrinkage
% hyperparameters fixed at the paper's values.
%
% See:
% Chan, J.C.C. (2023). Comparing stochastic volatility specifications for large
% Bayesian VARs, Journal of Econometrics, 235(2), 1419-1446.

run(fullfile(fileparts(fileparts(mfilename('fullpath'))),'setup.m'))

rng(20260916, 'twister')
fprintf('\n=== ex07: a VAR with factor stochastic volatility ===\n');

%% ------------------------------------------------------------------
%  1. Simulate, with the truth kept
%  ------------------------------------------------------------------
n = 8; r = 2; p = 1; T = 300;
k = 1 + n*p;
n0 = 4 + p;                                  % presample rows: 4 for the AR(4) fits, p for the lags

    % VAR coefficients: own first lag .6, small cross-lags, scaled to be stable
A_true = [zeros(1,n); .6*eye(n) + .08*randn(n)];
while max(abs(eig(A_true(2:end,:)'))) > .95
    A_true(2:end,:) = .95*A_true(2:end,:)/max(abs(eig(A_true(2:end,:)')));
end
L_true = .8*randn(n, r);                     % ones on the diagonal, zeros above it,
L_true(1:r,1:r) = tril(L_true(1:r,1:r), -1); % everything below free
L_true(sub2ind([n r], 1:r, 1:r)) = 1;
mu_true = [log(.2)*ones(n,1); zeros(r,1)];   % factor log-vols centered at zero
phi_true = .95*ones(n+r,1);
sig2_true = .05*ones(n+r,1);

Tall = T + n0 + 200;                         % 200 extra rows, discarded, to start from the stationary law
h = zeros(Tall, n+r);
h(1,:) = mu_true' + sqrt(sig2_true./(1-phi_true.^2))'.*randn(1,n+r);
for t = 2:Tall
    h(t,:) = mu_true' + phi_true'.*(h(t-1,:) - mu_true') + sqrt(sig2_true)'.*randn(1,n+r);
end
F_all = sqrt(exp(h(:, n+1:end))).*randn(Tall, r);
U_all = sqrt(exp(h(:, 1:n))).*randn(Tall, n);
Yall = zeros(Tall, n);
for t = 2:Tall
    Yall(t,:) = [1, Yall(t-1,:)]*A_true + F_all(t,:)*L_true' + U_all(t,:);
end
Yall = Yall(201:end, :);                     % drop the simulation burn-in
h_true = h(201:end, :); F_true = F_all(201:end, :);
Y0 = Yall(1:n0, :);   Y = Yall(n0+1:end, :);
h_true = h_true(n0+1:end, :); F_true = F_true(n0+1:end, :);
[~, X] = bvar.util.build_lags([Y0(end-p+1:end,:); Y], p);

fprintf('\nn = %d variables, r = %d factors, p = %d lag, T = %d\n', n, r, p, T);
fprintf('Sigma_t has %d free elements at each date; the model drives it with\n', n*(n+1)/2);
fprintf('%d log-volatility paths and %d free loadings\n', n+r, n*r - r*(r+1)/2);

%% ------------------------------------------------------------------
%  2. Prior and settings, at the values of the paper's preset
%  ------------------------------------------------------------------
nsim = 2000; burnin = 500;
kappa1 = .2^2; kappa2 = (.2^2)^2; kappa3 = 100;      % own lag, cross lag, intercept
Hyper.nuh = 3*ones(n+r,1);
Hyper.Sh = .1*(Hyper.nuh - 1);
Hyper.mu0 = zeros(n+r,1);   Hyper.Vmu = 100*ones(n+r,1);
Hyper.phi0 = .98*ones(n+r,1); Hyper.Vphi = .05^2*ones(n+r,1);
Hyper.l0 = 0; Hyper.Vl = 1;                          % prior on each free loading
[alp0, Valp] = bvar.priors.minn(p, kappa1, kappa2, kappa3, Y0, Y, 4);
phi_bnd = .998;                                      % the ml_varsv truncation bound
sv_offset = 1e-4;

%% ------------------------------------------------------------------
%  3. The sampler: factors, then coefficients and loadings, then the
%     log-volatilities and their parameters
%  ------------------------------------------------------------------
A = (X'*X + speye(k))\(X'*Y);
L = [eye(r); ones(n-r, r)];
varY = var(Y)';
mu = [log(varY/2); log(mean(varY))*ones(r,1)];
phi = .9 + .09*rand(n+r,1);
sig2 = .05 + .01*rand(n+r,1);
Uy = Y - X*A;
h = [zeros(T,n), repmat(mu(n+1:end)', T, 1)];
for ii = 1:n
    h(:,ii) = bvar.sv.init_approx1N(Uy(:,ii).^2, mu(ii), phi(ii), sig2(ii));
end

L_idx = find(tril(ones(n,r), -1) ~= 0);
idx_t = round(linspace(1, T, 30));            % dates at which every element is stored
tri = tril(true(n));
store_l = zeros(nsim, numel(L_idx));
store_F = zeros(nsim, T, r);
store_sig_sub = zeros(nsim, numel(idx_t), sum(tri(:)));
store_v1 = zeros(nsim, T);                    % variance of variable 1
store_c12 = zeros(nsim, T);                   % correlation of variables 1 and 2

fprintf('\ndrawing %d sweeps after %d burn-in...\n', nsim, burnin);
t0 = tic;
for isim = 1:nsim + burnin
    F = bvar.samplers.factor_fsv(Y, X, A, L, h);
    [A, L] = bvar.samplers.eq_fsv_load(Y, X, F, h, A, L, Valp, alp0, Hyper.Vl, Hyper.l0);

    ystar = log([Y - X*A - F*L', F].^2 + sv_offset);
    for jj = 1:n+r
        h(:,jj) = bvar.sv.ksc_ar1_mean(ystar(:,jj), h(:,jj), mu(jj), phi(jj), sig2(jj));
    end
    [mu, phi, sig2] = bvar.sv.sv_params(h, mu, phi, Hyper, phi_bnd);

    if isim > burnin
        isave = isim - burnin;
        store_l(isave,:) = L(L_idx)';
        store_F(isave,:,:) = F;
        D = exp(h(:,1:n)); G = exp(h(:,n+1:end));
        v1 = G*(L(1,:).^2)' + D(:,1);
        v2 = G*(L(2,:).^2)' + D(:,2);
        c12 = (G*(L(1,:).*L(2,:))')./sqrt(v1.*v2);
        store_v1(isave,:) = v1'; store_c12(isave,:) = c12';
        for it = 1:numel(idx_t)
            S = L*diag(G(idx_t(it),:))*L' + diag(D(idx_t(it),:));
            store_sig_sub(isave,it,:) = S(tri)';
        end
    end
end
fprintf('%.0f s\n', toc(t0));

%% ------------------------------------------------------------------
%  4. What the sampler recovered
%  ------------------------------------------------------------------
F_hat = squeeze(mean(store_F));
fprintf('\ncorrelation between the posterior mean factors and the true ones:');
fprintf(' %.3f', diag(corr(F_hat, F_true)));
fprintf('\n');

l_hat = mean(store_l)'; l_true = L_true(L_idx);
fprintf('\nthe %d free loadings, true against posterior mean\n', numel(l_true));
for j = 1:numel(l_true)
    fprintf('  L(%d,%d)  %+6.2f  %+6.2f\n', 1+mod(L_idx(j)-1,n), 1+floor((L_idx(j)-1)/n), l_true(j), l_hat(j));
end

    % coverage of the truth by 68% bands, over every element of Sigma_t at the
    % stored dates
Sig_true_sub = zeros(numel(idx_t), sum(tri(:)));
for it = 1:numel(idx_t)
    S = L_true*diag(exp(h_true(idx_t(it), n+1:end)))*L_true' + diag(exp(h_true(idx_t(it), 1:n)));
    Sig_true_sub(it,:) = S(tri)';
end
lo = squeeze(quantile(store_sig_sub, .16, 1));
hi = squeeze(quantile(store_sig_sub, .84, 1));
cov68 = mean(Sig_true_sub(:) >= lo(:) & Sig_true_sub(:) <= hi(:));
fprintf('\n68%% bands cover the truth in %.0f%% of the %d element-dates of Sigma_t\n', ...
    100*cov68, numel(Sig_true_sub));

%% ------------------------------------------------------------------
%  5. Figures: the factors, and two paths from Sigma_t
%  ------------------------------------------------------------------
v1_true = exp(h_true(:,n+1:end))*(L_true(1,:).^2)' + exp(h_true(:,1));
v2_true = exp(h_true(:,n+1:end))*(L_true(2,:).^2)' + exp(h_true(:,2));
c12_true = (exp(h_true(:,n+1:end))*(L_true(1,:).*L_true(2,:))')./sqrt(v1_true.*v2_true);

figure('Name', 'ex07: factor stochastic volatility');
for j = 1:r
    subplot(2,2,j); hold on; box off
    plot(F_true(:,j), 'Color', [.6 .6 .6]); plot(F_hat(:,j), 'k', 'LineWidth', 1);
    title(sprintf('factor %d: truth in grey, posterior mean in black', j)); hold off
end
subplot(2,2,3); hold on; box off
q = quantile(store_v1, [.16 .84]);
hb = bvar.util.shaded_band((1:T)', q(1,:)', q(2,:)');
hm = plot(mean(store_v1), 'k', 'LineWidth', 1); ht = plot(v1_true, 'r');
title('variance of variable 1'); hold off
legend([hb hm ht], {'68% credible band', 'posterior mean', 'truth'}, 'Location', 'northwest'); legend boxoff
subplot(2,2,4); hold on; box off
q = quantile(store_c12, [.16 .84]);
bvar.util.shaded_band((1:T)', q(1,:)', q(2,:)');
plot(mean(store_c12), 'k', 'LineWidth', 1); plot(c12_true, 'r');
title('correlation of variables 1 and 2'); hold off
drawnow
