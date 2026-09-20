%% your_data - does modeling the volatility improve forecasts on your data?
%
% Set the file, the columns, the lag length, the evaluation window and the chain
% lengths below, then run the script. At every origin it re-estimates three
% reduced-form VARs on the sample to that date - a homoskedastic one, one with a
% common volatility factor, and one with a volatility process per equation under
% the order-invariant impact matrix - forecasts them to the longest horizon in
% hs, and scores the forecasts by RMSFE and by the log predictive likelihood. It
% then plots the running difference in log score and writes the comparison to a
% csv and a mat file. None of the three models depends on the order of the
% columns.
%
% The defaults use the five series of the tutorial over a short window with short
% chains, which takes about a minute; the tutorial itself uses 140 origins and
% 5,000 draws after 1,000 burn-in.
%
% Give cols as column names, read with readtable. The selected columns must have
% no missing values inside the sample; rows missing at either end are dropped.
% The series must be stationary, since the prior means are zero.

tdir = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(tdir));
run(fullfile(repo, 'setup.m'))
lastwarn('');       % the report records the last warning raised from here on

%% ---- settings ----
file    = fullfile(repo, 'tutorials', 'sv_specification', 'macro5_Q.csv');
cols    = ["UNRATE" "PCECTPI" "FEDFUNDS" "NFCI" "GDPC1"];
datecol = "Date";                    % the column holding the dates; "" to skip the check
p       = 4;        % lags
n0      = 8;        % rows used as initial conditions
hs      = [1 4];    % horizons scored
nfirst  = 24;       % origins, counted back from the end of the sample
nsim    = 1000;     % draws kept per model and origin
burnin  = 200;      % draws discarded first
seed    = 1;
outdir  = tempdir;  % where the report goes; '' for none

%% ---- data ----
raw = readtable(file, 'VariableNamingRule', 'preserve');
sel = raw{:, cols};
keep = all(isfinite(sel), 2);
assert(any(keep), 'every row of the selected columns has a missing value');
lo = find(keep, 1);  hi = find(keep, 1, 'last');
assert(all(keep(lo:hi)), 'the selected columns have missing values inside the sample');
data = sel(lo:hi, :);
flat = std(data) == 0;
assert(~any(flat), 'these series are constant: %s', strjoin(cols(flat), ', '));
assert(n0 >= max(p, 4), 'n0 must be at least max(p,4)');
dates = [];
if strlength(datecol) > 0, dates = raw{lo:hi, datecol}; end
[nobs, n] = size(data);
k = 1 + n*p;
H = max(hs);
origins = (nobs - nfirst):(nobs - 1);
assert(origins(1) > n0 + p + 10, 'not enough data before the first origin');
no = numel(origins);
if isdatetime(dates)
    odate = dates(origins);
    fprintf('\n%d variables, %d origins (%s to %s), %d draws after %d burn-in\n', ...
        n, no, datestr(odate(1),'yyyyQQ'), datestr(odate(end),'yyyyQQ'), nsim, burnin);
else
    odate = origins(:);
    fprintf('\n%d variables, %d origins (rows %d to %d), %d draws after %d burn-in\n', ...
        n, no, origins(1), origins(end), nsim, burnin);
end

mname = ["homoskedastic" "VAR-CSV" "VAR-SV"];
nm = numel(mname);
point = nan(no, n, numel(hs), nm);
ljnt = nan(no, numel(hs), nm);
actual = nan(no, n, numel(hs));

t0 = tic;
for io = 1:no
    t = origins(io);
    Y0 = data(1:n0,:);  Y = data(n0+1:t,:);
    ylag = data(t:-1:t-p+1, :)';
    yobs = data(t+1:min(t+H, nobs), :);
    for ih = 1:numel(hs)
        if t + hs(ih) <= nobs, actual(io,:,ih) = data(t+hs(ih), :); end
    end
    cfg = struct('ylag', ylag, 'H', H, 'yobs', yobs);

        % homoskedastic, drawn directly from the natural conjugate posterior
    [~, X] = bvar.util.build_lags([Y0(end-p+1:end,:); Y], p);
    T = size(Y,1);
    [A0, VA0, nu0, S0] = bvar.priors.niw(p, [.2^2 100], Y0, Y, 'mlvarsv_ncp');
    iVA0 = sparse(1:k,1:k,1./VA0);
    KA = iVA0 + X'*X;  CKA = chol(KA,'lower');
    Ahat = (CKA')\(CKA\(sparse(1:k,1:k,VA0)\A0 + X'*Y));
    Shat = S0 + A0'*iVA0*A0 + Y'*Y - Ahat'*KA*Ahat;  Shat = (Shat+Shat')/2;
    rng(seed + io, 'twister');
    yh = zeros(nsim, H, n);  lj = zeros(nsim, H);
    for d = 1:nsim
        Sig = iwishrnd(Shat, nu0 + T);
        A = Ahat + (CKA'\randn(k,n))*chol(Sig,'lower')';
        [y1, ~, j1] = bvar.forecast.simulate('gauss', struct('A',A,'Sig',Sig), cfg);
        yh(d,:,:) = y1;  lj(d,:) = j1';
    end
    [point, ljnt] = keepscores(point, ljnt, io, 1, yh, lj, hs, nobs, t);

        % common stochastic volatility
    c = bvar.models.var_csv(Y0, Y, p, 'nsim', nsim, 'burnin', burnin, ...
        'seed', seed + io, 'draws', true);
    yh = zeros(nsim, H, n);  lj = zeros(nsim, H);
    for d = 1:nsim
        dr = struct('A', reshape(c.draws.A(d,:), k, n), 'Sig', reshape(c.draws.Sig(d,:), n, n), ...
            'h_T', c.draws.h(d,end), 'phi', c.draws.phi(d), 'sigh2', c.draws.sigh2(d));
        [y1, ~, j1] = bvar.forecast.simulate('csv', dr, cfg);
        yh(d,:,:) = y1;  lj(d,:) = j1';
    end
    [point, ljnt] = keepscores(point, ljnt, io, 2, yh, lj, hs, nobs, t);

        % one volatility per equation, order invariant
    s = bvar.models.var_sv(Y0, Y, p, 'model', 'OI', 'nsim', nsim, 'burnin', burnin, ...
        'seed', seed + io, 'draws', true);
    yh = zeros(nsim, H, n);  lj = zeros(nsim, H);
    for d = 1:nsim
        dr = struct('A', reshape(s.draws.A(d,:), k, n), ...
            'impact', reshape(s.draws.impact(d,:), n, n), 'h_T', s.draws.h_T(d,:), ...
            'phi', s.draws.phi(d,:), 'sig2', s.draws.sig2(d,:));
        [y1, ~, j1] = bvar.forecast.simulate('oisv', dr, cfg);
        yh(d,:,:) = y1;  lj(d,:) = j1';
    end
    [point, ljnt] = keepscores(point, ljnt, io, 3, yh, lj, hs, nobs, t);
end
fprintf('%d origins in %.0f seconds\n', no, toc(t0));

%% ---- the comparison ----
fprintf('\naccuracy relative to the %s VAR\n', mname(1));
fprintf('%-16s %8s %20s %20s\n', '', 'horizon', 'RMSFE gain, percent', 'log score gain');
gain = nan(nm, numel(hs));  lsg = nan(nm, numel(hs));
for im = 2:nm
    for ih = 1:numel(hs)
        ok = ~isnan(actual(:,1,ih));
        e0 = squeeze(point(ok,:,ih,1)) - actual(ok,:,ih);
        em = squeeze(point(ok,:,ih,im)) - actual(ok,:,ih);
        gain(im,ih) = median(100*(1 - sqrt(mean(em.^2,1))./sqrt(mean(e0.^2,1))));
        lsg(im,ih) = mean(ljnt(ok,ih,im) - ljnt(ok,ih,1));
        fprintf('%-16s %8d %20.1f %20.3f\n', mname(im), hs(ih), gain(im,ih), lsg(im,ih));
    end
end
fprintf(['the RMSFE column is the median gain across the variables; the log score\n' ...
    'column is the mean gain in the joint log predictive likelihood per origin\n']);
if no < 40
    fprintf('with only %d origins these averages are noisy: raise nfirst before reading much in\n', no);
end

figure;
hold on
for im = 2:nm
    ok = ~isnan(ljnt(:,1,im));
    plot(odate(ok), cumsum(ljnt(ok,1,im) - ljnt(ok,1,1)), 'LineWidth', 1.2);
end
yline(0, 'k:'); hold off; box off
legend(mname(2:nm), 'Location', 'best', 'Box', 'off');
ylabel('cumulative log score difference');
title(sprintf('Density forecasts against the %s VAR, %d step ahead', mname(1), hs(1)));

%% ---- the report ----
if ~isempty(outdir)
    [im, ih] = ndgrid(1:nm, 1:numel(hs));
    scores = table(mname(im(:))', reshape(hs(ih), [], 1), ...
        reshape([nan(1,numel(hs)); gain(2:end,:)]', [], 1), ...
        reshape([nan(1,numel(hs)); lsg(2:end,:)]', [], 1), ...
        'VariableNames', {'model','horizon','rmsfe_gain_percent','log_score_gain'});
    [iov, ivv, ihv] = ndgrid(1:no, 1:n, 1:numel(hs));
    fc = table();
    for m = 1:nm
        pm = point(:,:,:,m);
        fc = [fc; table(repmat(mname(m), numel(iov), 1), iov(:), cols(ivv(:))', ...
            reshape(hs(ihv), [], 1), pm(:), reshape(actual, [], 1), ...
            'VariableNames', {'model','origin','variable','horizon','forecast','actual'})]; %#ok<AGROW>
    end
    meta = struct('file', file, 'columns', cols, 'p', p, 'n0', n0, 'n', n, ...
        'origins', no, 'first_origin', origins(1), 'nsim', nsim, 'burnin', burnin, ...
        'seed', seed, 'horizons', hs);
    bvar.util.report('forecasting_report', struct('scores', scores, 'forecasts', fc), ...
        meta, outdir);
end

%% -------------------------------------------------------------------------
function [point, ljnt] = keepscores(point, ljnt, io, im, yh, lj, hs, nobs, t)
% Average the draws into the point forecast and the log predictive likelihood at
% the horizons that have an outturn.
nd = size(yh, 1);
for ih = 1:numel(hs)
    h = hs(ih);
    if t + h > nobs, continue; end
    point(io,:,ih,im) = mean(yh(:,h,:), 1);
    ljnt(io,ih,im) = bvar.util.logsumexp(lj(:,h)) - log(nd);
end
end
