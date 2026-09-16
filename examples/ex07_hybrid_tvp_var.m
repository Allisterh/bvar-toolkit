%% ex07 - A hybrid time-varying parameter VAR
%
% BOOK: Chapter 13, Section 13.2, in Bayesian Macroeconometrics: Methods and
% Applications (Chapman & Hall/CRC, forthcoming), treats the TVP-VAR with
% stochastic volatility in reduced form. The hybrid model here is in the
% structural form of Chan (2023), so its time-varying coefficients are
% structural ones and do not correspond one to one with the book's.
%
% THE MODEL. Equation i regresses y_it on the lags and on the contemporaneous
% values of the variables ordered before it:
%
%       y_it = x_t' beta_it - (alpha_i1,t y_1t + ... + alpha_i,i-1,t y_i-1,t)
%              + exp(h_it/2) e_it,
%
% with random-walk log-volatility h_it. Each equation has two switches:
% gam_beta_i for whether beta_it is time-varying, and gam_alpha_i for whether
% alpha_it is. Each coefficient that varies follows a random walk. The switches
% are drawn with the state paths integrated out, so each equation is estimated
% in whichever of its four configurations the data support.
%
% DATA. Simulated with n = 3, p = 1, T = 300, each equation from a different
% configuration:
%
%       equation 1   beta constant (it has no alpha)
%       equation 2   beta time-varying: the intercept and the own lag shift
%                    smoothly mid-sample, by .4 and .5; alpha constant
%       equation 3   beta constant; both alphas shift smoothly mid-sample,
%                    by -.6 and .5
%
% The sampler is the one of replications/chan2023_jbes_hybtvp/run_all.m, with
% the shrinkage hyperparameters held at their initial values rather than drawn,
% and one prior changed: the time variation in each slope and impact
% coefficient has prior standard deviation .1, where the paper's preset uses
% .01, which makes a constant coefficient and a varying one easier to tell
% apart. 3000 draws after 500 burn-in, against the published 50,000 after 1,000.
%
% See:
% Chan, J.C.C. (2023). Large Hybrid Time-Varying Parameter VARs, Journal of
% Business and Economic Statistics, 41(3): 890-905.

run(fullfile(fileparts(fileparts(mfilename('fullpath'))),'setup.m'))

rng(20260917, 'twister')
fprintf('\n=== ex07: a hybrid time-varying parameter VAR ===\n');

%% ------------------------------------------------------------------
%  1. Simulate, with the truth kept
%  ------------------------------------------------------------------
n = 3; p = 1; T = 300;
kb = n*p + 1;                                % coefficients per equation: intercept, then lags
k_beta = n*kb; k_alp = n*(n-1)/2;
n0 = 4;                                      % presample rows, for the AR(4) residual variances
Tall = n0 + T;

    % structural coefficients, one row per date; beta blocks are [1, y1, y2, y3]
beta_true = zeros(Tall, k_beta);
alp_true = zeros(Tall, k_alp);               % order: alpha_21, alpha_31, alpha_32
beta_true(:, 1:kb)        = repmat([.1 .5 .1 0], Tall, 1);     % equation 1: constant
beta_true(:, 2*kb+1:3*kb) = repmat([0 .1 .1 .5], Tall, 1);     % equation 3: constant
alp_true(:, 1) = .4;                                           % equation 2: constant alpha

    % equation 2: the intercept and own lag shift smoothly mid-sample;
    % equation 3: both alphas do
s = 1./(1 + exp(-((1:Tall)' - Tall/2)/(Tall/12)));
beta_true(:, kb+1:2*kb) = [.4*s, .2*ones(Tall,1), .2 + .5*s, .1*ones(Tall,1)];
alp_true(:, 2) = -.3 - .6*s;
alp_true(:, 3) = .2 + .5*s;

    % log-volatility: persistent, around fixed levels
hbar = log([.3 .5 .4]);
h_true = zeros(Tall, n); h_true(1,:) = hbar;
for t = 2:Tall
    h_true(t,:) = hbar + .98*(h_true(t-1,:) - hbar) + sqrt(.01)*randn(1,n);
end

Yall = zeros(Tall, n);
ylag = zeros(1, n);
for t = 1:Tall
    x = [1 ylag];
    rhs = zeros(1, n);
    for ii = 1:n
        rhs(ii) = x*beta_true(t, (ii-1)*kb+1:ii*kb)' + exp(h_true(t,ii)/2)*randn;
    end
    y = zeros(1, n);                          % forward substitution through the impact matrix
    y(1) = rhs(1);
    y(2) = rhs(2) - alp_true(t,1)*y(1);
    y(3) = rhs(3) - alp_true(t,2)*y(1) - alp_true(t,3)*y(2);
    Yall(t,:) = y;
    ylag = y;
end
Y0 = Yall(1:n0, :); Y = Yall(n0+1:end, :);
beta_true = beta_true(n0+1:end, :); alp_true = alp_true(n0+1:end, :);
X2 = [ones(T,1), [Y0(end,:); Y(1:end-1,:)]];

fprintf('\nn = %d, p = %d, T = %d, each equation simulated from a different configuration\n', n, p, T);

%% ------------------------------------------------------------------
%  2. Prior and settings: the package's preset, except the scale of the time
%     variation
%  ------------------------------------------------------------------
nsim = 3000; burnin = 500;
kappa = [.4 .04^2];                          % own lag, cross lag: the preset's initial values
sig2 = bvar.priors.resid_var_allvars_ridge(Y0, Y);
[C, idx_kappa1, idx_kappa2] = bvar.priors.minnesota_C(n, p, sig2);
[Valp, Vbeta] = bvar.priors.vtheta(idx_kappa1, idx_kappa2, [kappa .2 1], C, sig2);
nuh0 = 3*ones(n,1); Sh0 = .1*(nuh0 - 1);
Vsigbeta = .1^2*ones(k_beta,1);              % preset: .01^2, and .1^2 for intercepts
Vsigalp = .1^2*ones(k_alp,1);                % preset: .01^2
ah = zeros(n,1); Vh = 10*ones(n,1);
ap = .5*ones(1,2); bp = .5*ones(1,2);

%% ------------------------------------------------------------------
%  3. The sampler
%  ------------------------------------------------------------------
gam = zeros(n,2);
beta0 = zeros(k_beta,1); alp0 = zeros(k_alp,1);
U = zeros(T,n);
for ii = 1:n                                  % equation-wise least squares to start
    Xi = [X2 -Y(:,1:ii-1)];
    theta0 = (Xi'*Xi)\(Xi'*Y(:,ii));
    beta0((ii-1)*kb+1:ii*kb) = theta0(1:kb);
    alp0((ii-1)*(ii-2)/2+1:ii*(ii-1)/2) = theta0(kb+1:end);
    U(:,ii) = Y(:,ii) - Xi*theta0;
end
Sigbeta = Vsigbeta; Sigalp = Vsigalp;
p0 = .5*ones(n,2);
h0 = mean(log(U.^2))'; Sigh = Sh0;
h = repmat(h0', T, 1);
for ii = 1:n
    h(:,ii) = bvar.sv.ksc_rw_h0(log(U(:,ii).^2 + 1e-4), h(:,ii), Sigh(ii), h0(ii));
end

store_gam = zeros(nsim, n, 2);
store_b2own = zeros(nsim, T);                 % equation 2's own lag
store_a31 = zeros(nsim, T);                   % equation 3's alpha_31
store_b3own = zeros(nsim, T);                 % equation 3's own lag, constant in the truth

fprintf('\ndrawing %d sweeps after %d burn-in...\n', nsim, burnin);
t0 = tic;
for isim = 1:nsim + burnin
    for ii = 1:n
        ki = kb + ii - 1;
        ia = (ii-1)*(ii-2)/2+1:ii*(ii-1)/2;
        ib = (ii-1)*kb+1:ii*kb;
        Xi = [X2 -Y(:,1:ii-1)];
        Yi = Y(:,ii);

            % the two switches, with the state path integrated out, then the path
        [gam(ii,:), thetai, Ui, tilde_thetai] = bvar.samplers.eq_hyb_tvp(Xi, ...
            [beta0(ib); alp0(ia)], [Sigbeta(ib); Sigalp(ia)], Yi, h(:,ii), p0(ii,:)', kb, false, gam(ii,:));
        Thetai = reshape(thetai, ki, T)';

            % log-volatility
        h(:,ii) = bvar.sv.ksc_rw_h0(log(Ui.^2), h(:,ii), Sigh(ii), h0(ii));

            % the constant part and the signed size of the time variation, jointly
        tilde_Thetai = reshape(tilde_thetai, ki, T)';
        Vmui = [Vbeta(ib); Valp(ia); Vsigbeta(ib); Vsigalp(ia)];
        Wi = [Xi Xi.*tilde_Thetai];
        WiSig = Wi'*sparse(1:T, 1:T, exp(-h(:,ii)));
        Kmui = sparse(1:2*ki, 1:2*ki, 1./Vmui) + WiSig*Wi;
        mui = Kmui\(WiSig*Yi) + chol(Kmui,'lower')'\randn(2*ki,1);
        beta0(ib) = mui(1:kb);
        alp0(ia) = mui(kb+1:ki);
        Sigbeta(ib) = mui(ki+1:ki+kb).^2;
        Sigalp(ia) = mui(ki+kb+1:end).^2;

        if isim > burnin
            if ii == 2, store_b2own(isim-burnin,:) = Thetai(:,3)'; end
            if ii == 3, store_a31(isim-burnin,:) = Thetai(:,kb+1)';
                        store_b3own(isim-burnin,:) = Thetai(:,4)'; end
        end
    end

        % initial log-volatilities, their innovation variances, the switch probabilities
    Kh0 = sparse(1:n, 1:n, 1./Sigh + 1./Vh);
    h0 = Kh0\(ah./Vh + h(1,:)'./Sigh) + chol(Kh0,'lower')'\randn(n,1);
    eh = h - [h0'; h(1:T-1,:)];
    Sigh = 1./gamrnd(nuh0 + T/2, 1./(Sh0 + sum(eh.^2)'/2));
    p0 = betarnd(ap + gam, bp + 1 - gam);

    if isim > burnin
        store_gam(isim-burnin,:,:) = gam;
    end
end
fprintf('%.0f s\n', toc(t0));

%% ------------------------------------------------------------------
%  4. Which configuration each equation was estimated in
%  ------------------------------------------------------------------
true_cfg = [0 NaN; 1 0; 0 1];
fprintf('\nposterior probability of each configuration (beta, alpha), 1 = time-varying, 0 = constant\n');
fprintf('%-12s %8s %8s %8s %8s   true\n', '', '(0,0)', '(0,1)', '(1,0)', '(1,1)');
for ii = 1:n
    g = squeeze(store_gam(:,ii,:));
    pr_cfg = [mean(g(:,1)==0 & g(:,2)==0), mean(g(:,1)==0 & g(:,2)==1), ...
              mean(g(:,1)==1 & g(:,2)==0), mean(g(:,1)==1 & g(:,2)==1)];
    if ii == 1
        fprintf('  equation 1 %8.2f %8s %8.2f %8s   (%d,-)  no alpha in equation 1\n', ...
            pr_cfg(1)+pr_cfg(2), '', pr_cfg(3)+pr_cfg(4), '', true_cfg(1,1));
    else
        fprintf('  equation %d %8.2f %8.2f %8.2f %8.2f   (%d,%d)\n', ii, pr_cfg, true_cfg(ii,:));
    end
end

%% ------------------------------------------------------------------
%  5. Figures: a coefficient that drifts, one that does not, and a drifting
%     impact element
%  ------------------------------------------------------------------
figure('Name', 'ex07: hybrid TVP-VAR');
panels = {store_b2own, beta_true(:, kb+3),  'Equation 2: coefficient on y_{2,t-1}, time-varying'; ...
          store_b3own, beta_true(:, 2*kb+4), 'Equation 3: coefficient on y_{3,t-1}, constant'; ...
          store_a31,   alp_true(:, 2),       'Equation 3: \alpha_{31}, time-varying'};
for ip = 1:3
    subplot(3,1,ip); hold on; box off
    q = quantile(panels{ip,1}, [.16 .84]);
    hb = bvar.util.shaded_band((1:T)', q(1,:)', q(2,:)');
    hm = plot(mean(panels{ip,1}), 'k', 'LineWidth', 1);
    ht = plot(panels{ip,2}, 'r', 'LineWidth', 1);
    title(panels{ip,3}); hold off
    if ip == 1
        legend([hb hm ht], {'68% credible band', 'posterior mean', 'truth'}, 'Location', 'northwest');
    end
end
drawnow
