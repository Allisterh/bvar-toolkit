%% forecast_now - forecast the quarters after the end of the sample
%
% The evaluation in build.m and your_data.m stops before the last observation,
% so that every forecast has a realized value to be scored against. This script
% does the other thing: it estimates through the final observation and reports
% the forecasts of the quarters that have not happened yet.
%
% For each model it prints the predictive mean, the 68 and 90 per cent intervals
% and the probability of an event the settings name - by default that quarterly
% GDP growth is negative. Those come from the predictive distribution itself,
% which is an equally weighted mixture of normals, one per posterior draw:
% bvar.forecast.simulate returns each draw's mean and standard deviation,
% bvar.forecast.mixquantile inverts the mixture for the intervals, and the event
% probability is the mixture cdf at the threshold. Fitting one normal to the
% mixture, or taking quantiles of the draws' means, would both understate the
% uncertainty.
%
% Nothing here is scored, because there is nothing yet to score it against.
%
% Usage, from anywhere:  run tutorials/forecasting/forecast_now.m

tdir = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(tdir));
run(fullfile(repo, 'setup.m'))
lastwarn('');

%% ---- settings ----
file    = fullfile(repo, 'tutorials', 'sv_specification', 'macro5_Q.csv');
cols    = ["UNRATE" "PCECTPI" "FEDFUNDS" "NFCI" "GDPC1"];
vlabel  = ["Unemployment" "PCE inflation" "Fed funds" "NFCI" "GDP growth"];
datecol = "Date";
p       = 4;        % lags
n0      = 8;        % rows used as initial conditions
H       = 4;        % quarters ahead
nsim    = 5000;     % draws kept per model
burnin  = 1000;     % draws discarded first
seed    = 20260922;
event   = struct('variable', "GDPC1", 'below', 0);   % the event whose probability is reported
fanvar  = "GDPC1";  % the variable drawn as a fan chart
outdir  = tempdir;  % where the report goes; '' for none

%% ---- data ----
raw = readtable(file, 'VariableNamingRule', 'preserve');
sel = raw{:, cols};
keep = all(isfinite(sel), 2);
lo = find(keep, 1);  hi = find(keep, 1, 'last');
assert(all(keep(lo:hi)), 'the selected columns have missing values inside the sample');
data = sel(lo:hi, :);
dates = [];
if strlength(datecol) > 0, dates = raw{lo:hi, datecol}; end
[nobs, n] = size(data);
k = 1 + n*p;
Y0 = data(1:n0,:);  Y = data(n0+1:end,:);
ylag = data(end:-1:end-p+1, :)';
cfg = struct('ylag', ylag, 'H', H);       % no realized value: nothing to score
ievent = find(cols == event.variable);
ifan = find(cols == fanvar);
assert(~isempty(ievent) && ~isempty(ifan), 'the event or fan variable is not among cols');

if isdatetime(dates)
    qlab = string(dates(end) + calquarters(1:H), 'uuuuQQQ');
    fprintf('\nestimated through %s, forecasting %s to %s\n', ...
        string(dates(end),'uuuuQQQ'), qlab(1), qlab(end));
else
    qlab = "h = " + string(1:H);
    fprintf('\nestimated through observation %d, forecasting %d ahead\n', nobs, H);
end
fprintf('%d variables, %d observations after %d initial conditions, %d draws after %d burn-in\n', ...
    n, size(Y,1), n0, nsim, burnin);

%% ---- the three models, estimated through the last observation ----
mname = ["homoskedastic" "VAR-CSV" "VAR-OISV"];
nm = numel(mname);
MU = zeros(nsim, n, H, nm);      % each draw's conditional mean
SD = zeros(nsim, n, H, nm);      % and its standard deviation

    % homoskedastic, drawn directly from the natural conjugate posterior
[~, X] = bvar.util.build_lags([Y0(end-p+1:end,:); Y], p);
T = size(Y,1);
[A0, VA0, nu0, S0] = bvar.priors.niw(p, [.2^2 100], Y0, Y, 'mlvarsv_ncp');
iVA0 = sparse(1:k,1:k,1./VA0);
KA = iVA0 + X'*X;  CKA = chol(KA,'lower');
Ahat = (CKA')\(CKA\(sparse(1:k,1:k,VA0)\A0 + X'*Y));
Shat = S0 + A0'*iVA0*A0 + Y'*Y - Ahat'*KA*Ahat;  Shat = (Shat+Shat')/2;
rng(seed, 'twister');
for d = 1:nsim
    Sig = iwishrnd(Shat, nu0 + T);
    A = Ahat + (CKA'\randn(k,n))*chol(Sig,'lower')';
    [y1, ~, ~, s1] = bvar.forecast.simulate('gauss', struct('A',A,'Sig',Sig), cfg);
    MU(d,:,:,1) = y1';  SD(d,:,:,1) = s1';
end

c = bvar.models.var_csv(Y0, Y, p, 'nsim', nsim, 'burnin', burnin, 'seed', seed, 'draws', true);
for d = 1:nsim
    dr = struct('A', reshape(c.draws.A(d,:), k, n), 'Sig', reshape(c.draws.Sig(d,:), n, n), ...
        'h_T', c.draws.h(d,end), 'phi', c.draws.phi(d), 'sigh2', c.draws.sigh2(d));
    [y1, ~, ~, s1] = bvar.forecast.simulate('csv', dr, cfg);
    MU(d,:,:,2) = y1';  SD(d,:,:,2) = s1';
end

s = bvar.models.var_sv(Y0, Y, p, 'model', 'OI', 'nsim', nsim, 'burnin', burnin, ...
    'seed', seed, 'draws', true);
for d = 1:nsim
    dr = struct('A', reshape(s.draws.A(d,:), k, n), ...
        'impact', reshape(s.draws.impact(d,:), n, n), 'h_T', s.draws.h_T(d,:), ...
        'phi', s.draws.phi(d,:), 'sig2', s.draws.sig2(d,:));
    [y1, ~, ~, s1] = bvar.forecast.simulate('oisv', dr, cfg);
    MU(d,:,:,3) = y1';  SD(d,:,:,3) = s1';
end

%% ---- the predictive distribution, variable by variable ----
lv = [.05 .16 .50 .84 .95];
Q = zeros(n, H, nm, numel(lv));
pev = zeros(H, nm);
for im = 1:nm
    for j = 1:H
        for i = 1:n
            Q(i,j,im,:) = bvar.forecast.mixquantile(MU(:,i,j,im), SD(:,i,j,im), lv);
        end
        z = (event.below - MU(:,ievent,j,im))./SD(:,ievent,j,im);
        pev(j,im) = mean(normcdf(z));
    end
end

for im = 1:nm
    fprintf('\n%s\n', mname(im));
    fprintf('%-22s', 'variable');
    fprintf('%24s', qlab);
    fprintf('\n');
    for i = 1:n
        fprintf('%-22.22s', vlabel(i));
        for j = 1:H
            fprintf('%10.2f [%5.1f,%5.1f]', Q(i,j,im,3), Q(i,j,im,1), Q(i,j,im,5));
        end
        fprintf('\n');
    end
end
fprintf('\nposterior median with the 90%% interval in brackets\n');

fprintf('\nprobability that %s is below %g\n', event.variable, event.below);
fprintf('%-16s', '');  fprintf('%12s', qlab);  fprintf('\n');
for im = 1:nm
    fprintf('%-16s', mname(im));
    fprintf('%11.0f%%', 100*pev(:,im));
    fprintf('\n');
end
fprintf(['read from the predictive distribution itself, the mixture over draws;\n' ...
    'a single normal fitted to its mean and variance would give another number\n']);

%% ---- the fan chart ----
back = 12;
hist_idx = (nobs-back+1):nobs;
figure('Position', [100 100 780 340]);
if isdatetime(dates)
    xh = dates(hist_idx);  xf = dates(end) + calquarters(1:H)';
else
    xh = hist_idx(:);  xf = nobs + (1:H)';
end
hold on
bvar.util.shaded_band([xh(end); xf], [data(end,ifan); squeeze(Q(ifan,:,2,1))'], ...
    [data(end,ifan); squeeze(Q(ifan,:,2,5))'], .88);
bvar.util.shaded_band([xh(end); xf], [data(end,ifan); squeeze(Q(ifan,:,2,2))'], ...
    [data(end,ifan); squeeze(Q(ifan,:,2,4))'], .75);
plot(xh, data(hist_idx, ifan), 'k', 'LineWidth', 1.2);
plot([xh(end); xf], [data(end,ifan); squeeze(Q(ifan,:,2,3))'], 'k--', 'LineWidth', 1.2);
yline(0, 'k:');
hold off; box off
title(sprintf('%s: VAR-CSV forecast with 68%% and 90%% intervals', vlabel(ifan)));
exportgraphics(gcf, fullfile(tdir, 'fig_forecast_now.png'), 'Resolution', 150);

%% ---- the report ----
if ~isempty(outdir)
    rows = numel(mname)*n*H;
    T1 = table('Size', [rows 8], 'VariableTypes', ...
        {'string','string','string','double','double','double','double','double'}, ...
        'VariableNames', {'model','quarter','variable','mean','p05','p16','p84','p95'});
    r = 0;
    for im = 1:nm
        for j = 1:H
            for i = 1:n
                r = r + 1;
                T1(r,:) = {mname(im), qlab(j), cols(i), mean(MU(:,i,j,im)), ...
                    Q(i,j,im,1), Q(i,j,im,2), Q(i,j,im,4), Q(i,j,im,5)};
            end
        end
    end
    T2 = table('Size', [nm*H 4], 'VariableTypes', {'string','string','string','double'}, ...
        'VariableNames', {'model','quarter','event','probability'});
    r = 0;
    for im = 1:nm
        for j = 1:H
            r = r + 1;
            T2(r,:) = {mname(im), qlab(j), ...
                sprintf('%s below %g', event.variable, event.below), pev(j,im)};
        end
    end
    meta = struct('file', file, 'columns', cols, 'p', p, 'n0', n0, 'n', n, ...
        'T', size(Y,1), 'H', H, 'nsim', nsim, 'burnin', burnin, 'seed', seed, ...
        'estimated_through', string(dates(end), 'uuuuQQQ'), ...
        'event_variable', event.variable, 'event_below', event.below);
    bvar.util.report('forecast_now', struct('forecasts', T1, 'event', T2), meta, outdir);
end
