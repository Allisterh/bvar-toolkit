%% check_stability - do the reported quantities move when the chain is rerun?
%
% The build draws 5,000 times after a burn-in of 1,000 at each of 140 origins, and
% an inefficiency factor at the last origin measures how correlated those draws
% are. It does not establish whether the reported numbers would change under
% another chain. This script measures that directly.
%
% At four origins, two ordinary and two around the pandemic, each volatility model
% is estimated three times at the build's settings under different seeds, and once
% at four times the length. The quantities compared are those the tutorial
% reports: the joint log predictive likelihood at one and four quarters, and the
% predictive probability that GDP growth is negative one quarter ahead.
%
% This script is not part of the tutorial's results and is not run by the test
% suite. It takes about ten minutes.
%
% Usage, from anywhere:  run tutorials/forecasting/check_stability.m

tdir = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(tdir));
run(fullfile(repo, 'setup.m'));

p = 4;  n0 = 8;  H = 4;
short = struct('nsim', 5000, 'burnin', 1000);      % the build's settings
long = struct('nsim', 20000, 'burnin', 5000);
seeds = [11 22 33];

tbl = readtable(fullfile(repo, 'tutorials', 'sv_specification', 'macro5_Q.csv'), ...
    'VariableNamingRule', 'preserve');
vars = {'UNRATE','PCECTPI','FEDFUNDS','NFCI','GDPC1'};
data = tbl{:, vars};  dates = tbl.Date;
[nobs, n] = size(data);
k = 1 + n*p;
igdp = 5;

pick = [datetime(2005,10,1), datetime(2015,10,1), datetime(2019,10,1), datetime(2020,4,1)];
pname = ["2005Q4, ordinary" "2015Q4, ordinary" "2019Q4, before the pandemic" ...
         "2020Q2, in the pandemic"];
models = ["VAR-CSV" "VAR-OISV"];

for ic = 1:numel(pick)
    t = find(dates == pick(ic));
    Y0 = data(1:n0,:);  Y = data(n0+1:t,:);
    ylag = data(t:-1:t-p+1, :)';
    yobs = data(t+1:min(t+H, nobs), :);
    cfg = struct('ylag', ylag, 'H', H, 'yobs', yobs);
    fprintf('\n=== origin %s ===\n', pname(ic));
    fprintf('%-10s %-12s %6s %12s %12s %14s\n', 'model', 'chain', 'seed', ...
        'score h=1', 'score h=4', 'P(GDP<0) h=1');

    for im = 1:2
        runs = [repmat(short, 1, numel(seeds)), long];
        rseed = [seeds, seeds(1)];
        for ir = 1:numel(runs)
            cfgr = runs(ir);
            if im == 1
                r = bvar.models.var_csv(Y0, Y, p, 'nsim', cfgr.nsim, 'burnin', ...
                    cfgr.burnin, 'seed', rseed(ir), 'draws', true);
            else
                r = bvar.models.var_sv(Y0, Y, p, 'model', 'OI', 'nsim', cfgr.nsim, ...
                    'burnin', cfgr.burnin, 'seed', rseed(ir), 'draws', true);
            end
            nd = cfgr.nsim;
            lj = zeros(nd, H);  mu1 = zeros(nd, 1);  sd1 = zeros(nd, 1);
            for d = 1:nd
                if im == 1
                    dr = struct('A', reshape(r.draws.A(d,:), k, n), ...
                        'Sig', reshape(r.draws.Sig(d,:), n, n), 'h_T', r.draws.h(d,end), ...
                        'phi', r.draws.phi(d), 'sigh2', r.draws.sigh2(d));
                    spec = 'csv';
                else
                    dr = struct('A', reshape(r.draws.A(d,:), k, n), ...
                        'impact', reshape(r.draws.impact(d,:), n, n), ...
                        'h_T', r.draws.h_T(d,:), 'phi', r.draws.phi(d,:), ...
                        'sig2', r.draws.sig2(d,:));
                    spec = 'oisv';
                end
                [y1, ~, j1, s1] = bvar.forecast.simulate(spec, dr, cfg);
                lj(d,:) = j1';  mu1(d) = y1(1,igdp);  sd1(d) = s1(1,igdp);
            end
            pneg = mean(normcdf((0 - mu1)./sd1));
            lbl = sprintf('%d draws', cfgr.nsim);
            fprintf('%-10s %-12s %6d %12.3f %12.3f %13.1f%%\n', models(im), lbl, ...
                rseed(ir), bvar.util.logsumexp(lj(:,1)) - log(nd), ...
                bvar.util.logsumexp(lj(:,4)) - log(nd), 100*pneg);
        end
    end
end

fprintf(['\nthe three %d-draw rows of a block differ only in the seed, so their spread is\n' ...
    'what another run of the build would change. The %d-draw row is the reference:\n' ...
    'a short chain that sits inside the spread of its own seeds and close to it is long\n' ...
    'enough for the quantity in that column.\n'], short.nsim, long.nsim);
