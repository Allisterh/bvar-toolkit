% chan_eisenstat2018_jae_mltvpsv/preset - every constant the legacy estimation
% pipeline hard-codes (main_tvpsv.m dispatching to the ten workspace scripts TVPSV.m,
% TVP.m, TVP_R1_SV.m, TVP_R2_SV.m, TVP_R3_SV.m, VAR_SV.m, VAR.m, VAR_RS.m,
% VAR_RS_R1.m and VAR_RS_R2.m), one field per constant, each cited to its legacy
% source lines. Consumed by run_all.m in this folder; the legacy folder is never
% modified.
%
% The scripts write each prior as a scalar times ones(k,1); the scalar is stored
% here and run_all expands it once the dimension is known. A value that several
% scripts share is stored once, with every line that sets it. run_ml.m reads pr.ml.M;
% the DIC settings under pr.dic are recorded for reference, and no driver reads them.
%
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for
% time-varying parameter VARs with stochastic volatility, Journal of Applied
% Econometrics, 33(4), 509-532.

function pr = preset()

    % ---- data, run length and model switches (main_tvpsv.m) ----
pr.data_file  = 'USdata_2014Q4.xlsx';   % line 35; lives in legacy/, read-only
pr.data_range = 'B28:E269';             % line 35: 1954Q3-2014Q4, read with xlsread
pr.cols = [1 3 4];                      % line 36: GDP deflator growth, real GDP growth, Fed funds rate (line 38)
pr.n0 = 4;                              % line 38: Y0 = data(1:4,:); the rest is shortY (line 39)
pr.model_default  = 6;                  % line 19
pr.r_default      = 2;                  % line 22: regimes of the regime-switching models
pr.p_default      = 2;                  % line 23; the comment there allows p = 1,2,3,4 (p <= n0)
pr.nsims_default  = 20000;              % line 25
pr.burnin_default = 5000;               % line 26
pr.ml.M = 10000;                        % line 27: importance-sampling draws (run_ml.m)
pr.dic.nchains = 10;                    % line 28: chains behind each DIC estimate (not read)
pr.dic.simstep = 20;                    % line 32: thinning of the draws the DIC evaluates (not read)

    % ---- Gaussian priors: mean 0, variance 10 ----
pr.prior_mean = 0;      % atheta: TVPSV.m 11, TVP.m 12, VAR_SV.m 12, VAR.m 12, VAR_RS*.m 12;
                        % beta0/agam: TVP_R1_SV.m 12-13; gam0/abeta: TVP_R2_SV.m 12-13;
                        % beta0/gam0/amu: TVP_R3_SV.m 12-13, 15; ah: every SV script (TVPSV.m 12,
                        % TVP_R1_SV.m 14, TVP_R2_SV.m 14, TVP_R3_SV.m 14, VAR_SV.m 13)
pr.prior_var = 10;      % Vtheta, Vbeta, Vgam, Vmu and Vh at the same lines

    % ---- inverse-gamma priors on the variances: shape nu, scale S*(nu-1) ----
pr.nu = 5;              % nutheta0: TVPSV.m 13, TVP.m 13; nugam0: TVP_R1_SV.m 15;
                        % nubeta0: TVP_R2_SV.m 15; numu0: TVP_R3_SV.m 16; nuh0: TVPSV.m 16,
                        % TVP_R1_SV.m 16, TVP_R2_SV.m 18, TVP_R3_SV.m 17, VAR_SV.m 14;
                        % nu0: TVP.m 16, VAR.m 13, VAR_RS*.m 13
pr.S_state = .01^2;     % state innovations of the slopes and the impact elements:
                        % TVPSV.m 14, TVP.m 14, TVP_R1_SV.m 15, TVP_R2_SV.m 16
pr.S_intercept = .1^2;  % state innovations of the intercepts: TVPSV.m 15, TVP.m 15,
                        % TVP_R2_SV.m 17; and of mu_t in TVP-R3-SV: TVP_R3_SV.m 16
pr.S_h = .01;           % log-volatility innovations: TVPSV.m 16, TVP_R1_SV.m 16,
                        % TVP_R2_SV.m 18, TVP_R3_SV.m 17, VAR_SV.m 14
pr.S_sig = 1;           % error variances of the homoskedastic models: TVP.m 16, VAR.m 13,
                        % VAR_RS*.m 13 (written S0 = ones(n,1).*(nu0-1))

    % ---- starting values of the chains ----
pr.init.state_var = .01;    % Sigtheta: TVPSV.m 46, TVP.m 43; Siggam: TVP_R1_SV.m 53;
                            % Sigbeta: TVP_R2_SV.m 64
pr.init.sigh = .01;         % Sigh: TVPSV.m 47, TVP_R1_SV.m 54, TVP_R2_SV.m 65
pr.init.sigh_cvarsv = .05;  % Sigh: VAR_SV.m 41
pr.init.sigh_r3 = .1;       % Sigh: TVP_R3_SV.m 71
pr.init.sigmu_r3 = .1;      % Sigmu: TVP_R3_SV.m 70
pr.init.sig = 1;            % Sig: TVP.m 44, VAR.m 38
    % the log-volatilities start at log(var(shortY)) (TVPSV.m 48, TVP_R1_SV.m 55,
    % TVP_R2_SV.m 66, VAR_SV.m 42) or, in TVP-R3-SV, at the log mean squared
    % residual of the starting fit (TVP_R3_SV.m 72); the initial states theta0,
    % gam0 and beta0 at zero (TVPSV.m 50, TVP.m 45, TVP_R1_SV.m 57, TVP_R2_SV.m 68)
    % and mu0 at mean(shortY) (TVP_R3_SV.m 66); the constant coefficients at
    % least squares (TVP_R1_SV.m 52, TVP_R2_SV.m 63, TVP_R3_SV.m 68-69, VAR_RS.m 46,
    % VAR_RS_R1.m 40, VAR_RS_R2.m 46)

    % ---- shared sampler constants ----
pr.sv_offset = .0001;       % log(u.^2 + .0001): TVPSV.m 79, TVP_R1_SV.m 92, TVP_R2_SV.m 102,
                            % TVP_R3_SV.m 107, VAR_SV.m 62
pr.progress_every = 10000;  % loop-counter display: every script except TVP-R3-SV
pr.progress_every_r3 = 5000;    % TVP_R3_SV.m 136

    % ---- regime-switching VARs (VAR_RS.m, VAR_RS_R1.m, VAR_RS_R2.m) ----
pr.rs.alp0 = 2;             % line 14: symmetric Dirichlet prior on each row of P
pr.rs.P_stay = .8;          % line 37: starting P, .8 on the diagonal
pr.rs.P_move = .2;          % line 36: and .2/(r-1) off it (VAR_RS_R1.m 35-36,
                            % VAR_RS_R2.m 36-37); a literal, since 1 - .8 is a
                            % different double
pr.rs.p2_first = 1/3;       % line 59: p(s_1) in the filter, 1/3 for every r
                            % (VAR_RS_R1.m 59, VAR_RS_R2.m 60); the filter
                            % normalizes each step, so any common value gives
                            % the same draws
end
