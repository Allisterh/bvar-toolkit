% chan_eisenstat2018_jae_mltvpsv/run_ml - the cp_ml pipeline of main_tvpsv.m,
% functionized: the estimation (run_all in this folder) and then the log marginal
% likelihood of the model by importance sampling (the bvar.ml.mltvpsv_* routine for the
% model), in the legacy order and on one continuous rng stream.
%
%   out = run_ml(model, p, r, nsims, burnin, seed)
%   out = run_ml(..., 'M', 10000, 'bugcompat', false, 'data', D)
%
%   model .. seed - exactly as run_all; seed seeds rng once, before the estimation, and
%       the marginal likelihood continues the same stream
%   'M'         - importance draws, default 10000 (main_tvpsv.m line 27); rounded up to
%       a multiple of 20, the number of batches behind the numerical standard error
%   'bugcompat' - read by the three RS models only. false (default) starts the Hamilton
%       filter from p(s_1 = j) = 1/r; true reproduces the published likelihood
%       (intlike_var_rs.m), which starts from 1/3 whatever r is and so shifts every log
%       likelihood, and the log marginal likelihood, by log(r/3).
%   'data'      - passed to run_all
%
% Output: the run_all output plus out.lml, out.lmlstd, out.ml (store_w, the M log
% weights, and bigml, the 20 batch estimates) and out.M and out.bugcompat. The legacy
% displays 'Computing marginal likelihood of ...' and 'log marginal likelihood: ...'
% are reproduced; the routines print nothing.
%
% See:
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for time-varying
% parameter VARs with stochastic volatility, Journal of Applied Econometrics, 33(4),
% 509-532.

function out = run_ml(model, p, r, nsims, burnin, seed, varargin)
thisdir = fileparts(mfilename('fullpath'));
od = cd(thisdir);
guard = onCleanup(@() cd(od));
pr = preset();
clear guard

if nargin < 1, model = []; end
if nargin < 2, p = []; end
if nargin < 3, r = []; end
if nargin < 4, nsims = []; end
if nargin < 5, burnin = []; end
if nargin < 6, seed = []; end
M = pr.ml.M; bugcompat = false; pass = {};
if mod(numel(varargin), 2) ~= 0
    error('run_ml:badOption', 'options must come in name-value pairs');
end
for iv = 1:2:numel(varargin)
    switch lower(char(string(varargin{iv})))
        case 'm',         M = varargin{iv+1};
        case 'bugcompat', bugcompat = varargin{iv+1};
        case 'data',      pass = [pass, varargin(iv:iv+1)]; %#ok<AGROW>
        otherwise, error('run_ml:badOption', 'unknown option ''%s''', char(string(varargin{iv})));
    end
end
if ~(isnumeric(M) && isscalar(M) && M >= 1 && M == fix(M))
    error('run_ml:badOption', 'M must be a positive integer');
end
if ~(islogical(bugcompat) || isnumeric(bugcompat)) || ~isscalar(bugcompat)
    error('run_ml:badOption', 'bugcompat must be true or false');
end
bugcompat = logical(bugcompat);

out = run_all(model, p, r, nsims, burnin, seed, pass{:});

    % [main_tvpsv.m 44-84]; the routines' own displays [ml_*.m 11-12]
labels = {'TVP-SV', 'TVP', 'TVP-R1-SV', 'TVP-R2-SV', 'TVP-R3-SV', 'VAR-SV', 'VAR', ...
    'VAR-RS', 'VAR-RS-R1', 'VAR-RS-R2'};
disp(['Computing marginal likelihood of ' labels{out.model_num} '.... ']);
o = out;
switch out.model_num
    case 1
        [lml, lmlstd, mlout] = bvar.ml.mltvpsv_tvpsv(o.Y, o.store_Sigtheta, o.store_Sigh, ...
            o.store_h0, o.store_theta0, o.prior, o.bigX, M);
    case 2
        [lml, lmlstd, mlout] = bvar.ml.mltvpsv_tvp(o.Y, o.store_Sig, o.store_Sigtheta, ...
            o.store_theta0, o.prior, o.bigX, M);
    case 3
        [lml, lmlstd, mlout] = bvar.ml.mltvpsv_tvp_r1_sv(o.Y, o.store_beta, o.store_Siggam, ...
            o.store_Sigh, o.store_h0, o.store_gam0, o.prior, o.Xtilde, o.W, M);
    case 4
        [lml, lmlstd, mlout] = bvar.ml.mltvpsv_tvp_r2_sv(o.Y, o.store_gam, o.store_Sigbeta, ...
            o.store_Sigh, o.store_h0, o.store_beta0, o.prior, o.Xtilde, o.W, M);
    case 5
        [lml, lmlstd, mlout] = bvar.ml.mltvpsv_tvp_r3_sv(o.Y, o.store_beta, o.store_gam, ...
            o.store_Sigmu, o.store_Sigh, o.store_h0, o.store_mu0, o.prior, o.Z, M);
    case 6
        [lml, lmlstd, mlout] = bvar.ml.mltvpsv_cvarsv(o.Y, o.store_theta, o.store_Sigh, ...
            o.store_h0, o.prior, o.bigX, M);
    case 7
        [lml, lmlstd, mlout] = bvar.ml.mltvpsv_cvar(o.Y, o.store_theta, o.store_Sig, ...
            o.prior, o.bigX, M);
    case 8
        [lml, lmlstd, mlout] = bvar.ml.mltvpsv_rs(o.Y, o.store_theta, o.store_Sig, ...
            o.store_P, o.prior, o.bigX, M, bugcompat);
    case 9
        [lml, lmlstd, mlout] = bvar.ml.mltvpsv_rs_r1(o.Y, o.store_theta, o.store_Sig, ...
            o.store_P, o.prior, o.bigX, M, bugcompat);
    case 10
        [lml, lmlstd, mlout] = bvar.ml.mltvpsv_rs_r2(o.Y, o.store_theta, o.store_Sig, ...
            o.store_P, o.prior, o.bigX, M, bugcompat);
end
disp(' ')
fprintf('log marginal likelihood: %.1f (%.2f)\n', lml, lmlstd);
disp(' ' );

out.lml = lml; out.lmlstd = lmlstd; out.ml = mlout;
out.M = 20*ceil(M/20); out.bugcompat = bugcompat;
end
