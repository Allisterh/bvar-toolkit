%% bench_density - two ways to score a multi-step density, on the same draws
%
% Given a posterior draw of a VAR with common stochastic volatility, the log
% predictive likelihood of a realized value h quarters ahead can be estimated
% two ways.
%
%   CONDITIONAL GAUSSIAN, which bvar.forecast.simulate uses: simulate only the
%   volatility path, then evaluate the realized value against the exact Gaussian
%   that the path implies, with mean the VAR iterated forward and variance the
%   sum of Psi_i*Sigma_{T+h-i}*Psi_i' over i = 0 to h-1.
%
%   PATH BASED, written out below as path_score: simulate the data as well, and
%   evaluate the realized value against the one-step covariance at the state the
%   simulated path happens to reach.
%
% Both are unbiased estimators of the predictive DENSITY, and neither is an
% unbiased estimator of its logarithm: by Jensen's inequality the log of an
% unbiased density estimate sits below the log density on average, the more so
% the noisier the estimate. The two are compared here on the same posterior
% draws, at an ordinary realized value and at an extreme one, over several
% simulation sizes and seeds, to measure that gap.
%
% This script is not part of the tutorial's results and is not run by the test
% suite. It takes about a minute.
%
% Usage, from anywhere:  run tutorials/forecasting/bench_density.m

tdir = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(tdir));
run(fullfile(repo, 'setup.m'));

p = 4;  n0 = 8;  H = 4;  nsim = 5000;  burnin = 1000;
sizes = [250 1000 5000];
seeds = 1:5;

tbl = readtable(fullfile(repo, 'tutorials', 'sv_specification', 'macro5_Q.csv'), ...
    'VariableNamingRule', 'preserve');
vars = {'UNRATE','PCECTPI','FEDFUNDS','NFCI','GDPC1'};
data = tbl{:, vars};  dates = tbl.Date;
[nobs, n] = size(data);
k = 1 + n*p;

    % an ordinary stretch, and the one that contains the pandemic quarters
cases = [find(dates == datetime(2015,10,1)), find(dates == datetime(2019,10,1))];
cname = ["ordinary (2016Q1-Q4)" "pandemic (2020Q1-Q4)"];

for ic = 1:numel(cases)
    t = cases(ic);
    Y0 = data(1:n0,:);  Y = data(n0+1:t,:);
    ylag = data(t:-1:t-p+1, :)';
    yobs = data(t+1:t+H, :);
    cfg = struct('ylag', ylag, 'H', H, 'yobs', yobs);
    fprintf('\n=== %s, origin %s ===\n', cname(ic), datestr(dates(t), 'yyyyQQ'));
    fprintf('realized GDP growth: ');
    fprintf('%.1f ', yobs(:,5));
    fprintf('\n');

    c = bvar.models.var_csv(Y0, Y, p, 'nsim', nsim, 'burnin', burnin, 'seed', 42, 'draws', true);
    draws = cell(nsim, 1);
    for d = 1:nsim
        draws{d} = struct('A', reshape(c.draws.A(d,:), k, n), ...
            'Sig', reshape(c.draws.Sig(d,:), n, n), 'h_T', c.draws.h(d,end), ...
            'phi', c.draws.phi(d), 'sigh2', c.draws.sigh2(d));
    end

    for ih = [1 4]
        fprintf('\n  h = %d, joint log score over the five variables\n', ih);
        fprintf('  %8s %28s %28s\n', 'draws', 'conditional Gaussian', 'path based');
        fprintf('  %8s %14s %13s %14s %13s\n', '', 'mean', 'spread', 'mean', 'spread');
        for ns = sizes
            sc = zeros(numel(seeds), 2);
            for is = 1:numel(seeds)
                lg = zeros(ns, 1);  lp = zeros(ns, 1);
                rng(seeds(is), 'twister');
                for d = 1:ns
                    [~, ~, j1] = bvar.forecast.simulate('csv', draws{d}, cfg);
                    lg(d) = j1(ih);
                end
                rng(seeds(is), 'twister');
                for d = 1:ns
                    j2 = path_score(draws{d}, cfg);
                    lp(d) = j2(ih);
                end
                sc(is,1) = bvar.util.logsumexp(lg) - log(ns);
                sc(is,2) = bvar.util.logsumexp(lp) - log(ns);
            end
            fprintf('  %8d %14.2f %13.2f %14.2f %13.2f\n', ns, ...
                mean(sc(:,1)), max(sc(:,1)) - min(sc(:,1)), ...
                mean(sc(:,2)), max(sc(:,2)) - min(sc(:,2)));
        end
    end
end

fprintf(['\nspread is the range over %d seeds, the posterior draws held fixed, so it is\n' ...
    'simulation noise alone. At h = 1 the two estimators coincide, since no data path\n' ...
    'has been simulated yet. Beyond that the path-based estimator drifts down and its\n' ...
    'spread widens, and both effects are worst where the realized value is ' ...
    'extreme.\n'], numel(seeds));

%% -------------------------------------------------------------------------
function lj = path_score(dr, cfg)
% The estimator this tutorial does not use: simulate the data as well as the
% volatility, and score the realized value against the one-step covariance at
% the state the simulated path reaches.
n = size(dr.A, 2);
p = (size(dr.A,1) - 1)/n;
H = cfg.H;
lj = nan(H, 1);
h = dr.h_T;  phi = dr.phi;  sdh = sqrt(dr.sigh2);
CSig0 = chol(dr.Sig, 'lower');
x = [1, reshape(cfg.ylag, 1, [])];
for j = 1:H
    h = phi*h + sdh*randn;
    CS = exp(h/2)*CSig0;
    EY = x*dr.A;
    if size(cfg.yobs,1) >= j
        u = cfg.yobs(j,:) - EY;
        z = CS\u';
        lj(j) = -n/2*log(2*pi) - sum(log(diag(CS))) - .5*(z'*z);
    end
    x = [1, EY + (CS*randn(n,1))', x(2:end-n)];      %#ok<AGROW>
end
end
