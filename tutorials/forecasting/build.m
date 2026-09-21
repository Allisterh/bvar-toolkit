%% build - regenerate the figures and numbers of tutorials/forecasting/README.md
%
% A recursive, pseudo-out-of-sample comparison of three reduced-form BVARs on the
% five quarterly series of tutorials/sv_specification: a homoskedastic VAR, a VAR
% with one common volatility factor (VAR-CSV), and a VAR with one log-volatility
% per equation under the order-invariant impact matrix of Chan, Koop and Yu
% (2024) (VAR-SV). None of the three depends on the order of the columns, so the
% comparison never asks the reader to justify an ordering.
%
% At each origin every model is re-estimated on the sample to that date and
% forecast one to four quarters ahead, one simulated path per posterior draw
% through bvar.forecast.simulate. Point forecasts are scored by RMSFE and density
% forecasts by the log predictive likelihood, which is the log of the average of
% exp(log density) over the draws. The homoskedastic model is also scored exactly,
% through bvar.forecast.predictive, which measures the simulation noise in the
% other two.
%
% Density forecasts are also checked for calibration: how often the realized
% value fell inside the interval a model claimed, and how wide that interval
% was. An interval that is narrow and covers is worth more than one that is
% narrow.
%
% Forecasts are grouped by the quarter they are for.
%
% The per-origin scores go to scores_by_origin.mat and the figures are written
% next to this file. Everything printed goes to a log in tempdir.
%
% Usage, from anywhere:  run tutorials/forecasting/build.m

tdir = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(tdir));
logf = fullfile(tempdir, 'bvar_forecasting_build_log.txt');
if exist(logf, 'file'), delete(logf); end
diary(logf);
fprintf('tutorials/forecasting/build.m, %s, MATLAB %s\n', ...
    char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm')), version);
t_all = tic;
run(fullfile(repo, 'setup.m'));

%% ---- settings ----
p = 4;  n0 = 8;  H = 4;  hs = [1 4];
nsim = 5000;  burnin = 1000;
first_forecast = datetime(1990,1,1);
seed0 = 20260921;
mname = ["homoskedastic" "VAR-CSV" "VAR-OISV"];
NLc = newline;
nm = numel(mname);

%% ---- data, the panel of tutorials/sv_specification ----
tbl = readtable(fullfile(repo, 'tutorials', 'sv_specification', 'macro5_Q.csv'), ...
    'VariableNamingRule', 'preserve');
vars = {'UNRATE','PCECTPI','FEDFUNDS','NFCI','GDPC1'};
vlabel = ["Unemployment" "PCE inflation" "Fed funds" "NFCI" "GDP growth"];
data = tbl{:, vars};
dates = tbl.Date;
[nobs, n] = size(data);
k = 1 + n*p;
    % every origin has a realized value one quarter ahead
origins = find(dates >= first_forecast, 1) - 1 : nobs - 1;
no = numel(origins);
fprintf('\n%d variables, %d quarters (%s to %s)\n', n, nobs, ...
    datestr(dates(1),'yyyyQQ'), datestr(dates(end),'yyyyQQ'));
fprintf('%d forecast origins, %s to %s; %d draws after %d burn-in per model and origin\n', ...
    no, datestr(dates(origins(1)),'yyyyQQ'), datestr(dates(origins(end)),'yyyyQQ'), nsim, burnin);
fprintf('the models are re-estimated at every origin, on data up to that quarter only\n');

point = nan(no, n, 2, nm);        % point forecast, by origin, variable, horizon, model
lpl = nan(no, n, 2, nm);          % log predictive likelihood, per variable
ljnt = nan(no, 2, nm);            % joint log predictive likelihood
psd = nan(no, n, 2, nm);          % predictive standard deviation
actual = nan(no, n, 2);
lpl_exact = nan(no, n, 2);        % the homoskedastic model scored without simulation
pit = nan(no, n, 2, nm);          % the predictive cdf at the realized value
w80 = nan(no, n, 2, nm);          % width of the 80 per cent interval
w95 = nan(no, n, 2, nm);          % and of the 95 per cent one

for io = 1:no
    t = origins(io);
    Y0 = data(1:n0, :);  Y = data(n0+1:t, :);
    T = size(Y, 1);
    ylag = data(t:-1:t-p+1, :)';
    hmax = min(H, nobs - t);
    yobs = data(t+1:t+hmax, :);
    for ih = 1:2
        if t + hs(ih) <= nobs, actual(io,:,ih) = data(t+hs(ih), :); end
    end
    cfg = struct('ylag', ylag, 'H', H, 'yobs', yobs);

        % ---- model 1: homoskedastic, natural conjugate ----
    [~, X] = bvar.util.build_lags([Y0(end-p+1:end,:); Y], p);
    [A0, VA0, nu0, S0] = bvar.priors.niw(p, [.2^2 100], Y0, Y, 'mlvarsv_ncp');
    iVA0 = sparse(1:k, 1:k, 1./VA0);
    KA = iVA0 + X'*X;
    CKA = chol(KA, 'lower');
    Ahat = (CKA')\(CKA\(sparse(1:k,1:k,VA0)\A0 + X'*Y));
    Shat = S0 + A0'*iVA0*A0 + Y'*Y - Ahat'*KA*Ahat;
    Shat = (Shat + Shat')/2;
    rng(seed0 + io, 'twister');
    Ad = zeros(nsim, k*n);  Sd = zeros(nsim, n, n);
    for d = 1:nsim
        Sig = iwishrnd(Shat, nu0 + T);
        A = Ahat + (CKA'\randn(k,n))*chol(Sig,'lower')';
        Ad(d,:) = A(:)';  Sd(d,:,:) = Sig;
    end
    [yh, ld, lj, sd] = run_draws('gauss', Ad, Sd, [], cfg, n, k, H);
    [point, lpl, ljnt, psd, pit, w80, w95] = store(point, lpl, ljnt, psd, pit, w80, w95, io, 1, yh, ld, lj, sd, hs, nobs, t, squeeze(actual(io,:,:)));

        % the same model scored exactly, with no simulation
    [mu, sdx] = bvar.forecast.predictive(Ad, Sd, ylag, H);
    for ih = 1:2
        if t + hs(ih) > nobs, continue; end
        z = (actual(io,:,ih) - mu(:,:,hs(ih)))./sdx(:,:,hs(ih));
        ldx = -.5*log(2*pi*sdx(:,:,hs(ih)).^2) - .5*z.^2;
        lpl_exact(io,:,ih) = bvar.util.logsumexp(ldx) - log(nsim);
    end

        % ---- model 2: VAR-CSV ----
    c = bvar.models.var_csv(Y0, Y, p, 'nsim', nsim, 'burnin', burnin, ...
        'seed', seed0 + io, 'draws', true);
    [yh, ld, lj, sd] = run_draws('csv', c.draws.A, c.draws.Sig, c.draws, cfg, n, k, H);
    [point, lpl, ljnt, psd, pit, w80, w95] = store(point, lpl, ljnt, psd, pit, w80, w95, io, 2, yh, ld, lj, sd, hs, nobs, t, squeeze(actual(io,:,:)));

        % ---- model 3: VAR-SV, order invariant ----
    s = bvar.models.var_sv(Y0, Y, p, 'model', 'OI', 'nsim', nsim, 'burnin', burnin, ...
        'seed', seed0 + io, 'draws', true);
    [yh, ld, lj, sd] = run_draws('oisv', s.draws.A, s.draws.impact, s.draws, cfg, n, k, H);
    [point, lpl, ljnt, psd, pit, w80, w95] = store(point, lpl, ljnt, psd, pit, w80, w95, io, 3, yh, ld, lj, sd, hs, nobs, t, squeeze(actual(io,:,:)));

    if mod(io, 20) == 0
        fprintf('  %d of %d origins, %.1f minutes\n', io, no, toc(t_all)/60);
    end
    if io == no                       % how well the two chains mix at the longest sample
        IFc = bvar.diag.inefficiency_factor([c.draws.phi, c.draws.sigh2, c.draws.Sig(:,1)], 200);
        IFs = bvar.diag.inefficiency_factor([s.draws.phi(:,1), s.draws.sig2(:,1), s.draws.h_T(:,1)], 200);
    end
end

%% ---- the scores ----
fprintf('%s=== Accuracy relative to the homoskedastic VAR ===%s', NLc, NLc);
odate = dates(origins);
tdate = NaT(no, 2);                      % the quarter each forecast is FOR
for ih = 1:2
    tt = origins(:) + hs(ih);
    okt = tt <= nobs;
    tdate(okt, ih) = dates(tt(okt));
end
covid = datetime(2020,1,1);
bname = ["all" "targets through 2019" "targets 2020 onwards"];

fprintf(['forecasts are grouped by the quarter they are FOR, so a four-quarter-ahead%s' ...
    'forecast made in 2019Q4 counts as a forecast of 2020Q4%s'], NLc, NLc);
fprintf('%s%-16s %7s %22s %14s %14s%s', NLc, '', 'horizon', 'group', ...
    'RMSFE gain', 'log score gain', NLc);
for im = 2:nm
    for ih = 1:2
        for ib = 1:3
            switch ib
                case 1, sel = true(no,1);
                case 2, sel = tdate(:,ih) < covid;
                case 3, sel = tdate(:,ih) >= covid;
            end
            ok = sel & ~isnan(actual(:,1,ih));
            if ~any(ok), continue; end
            e0 = squeeze(point(ok,:,ih,1)) - actual(ok,:,ih);
            em = squeeze(point(ok,:,ih,im)) - actual(ok,:,ih);
            g = median(100*(1 - sqrt(mean(em.^2,1))./sqrt(mean(e0.^2,1))));
            dj = mean(ljnt(ok,ih,im) - ljnt(ok,ih,1));
            lbl = '';
            if ib == 1, lbl = char(mname(im)); end
            fprintf('%-16s %7d %22s %10.1f%% %14.3f   (%d forecasts)%s', lbl, hs(ih), ...
                bname(ib), g, dj, nnz(ok), NLc);
        end
    end
end
fprintf(['%sthe RMSFE column is the median gain across the five variables and the log%s' ...
    'score column the mean gain in the joint log predictive likelihood, per quarter%s'], ...
    NLc, NLc, NLc);

fprintf('%sby variable, against the homoskedastic VAR%s', NLc, NLc);
for ih = 1:2
    ok = ~isnan(actual(:,1,ih));
    fprintf('%s  h = %d, %d forecasts%s', NLc, hs(ih), nnz(ok), NLc);
    fprintf('  %-22s', 'variable');
    for im = 2:nm
        fprintf(' %15s %15s', mname(im) + " RMSFE", mname(im) + " score");
    end
    fprintf('%s', NLc);
    for i = 1:n
        fprintf('  %-22.22s', vlabel(i));
        for im = 2:nm
            e0 = point(ok,i,ih,1) - actual(ok,i,ih);
            em = point(ok,i,ih,im) - actual(ok,i,ih);
            fprintf(' %14.1f%% %15.3f', 100*(1 - sqrt(mean(em.^2))/sqrt(mean(e0.^2))), ...
                mean(lpl(ok,i,ih,im) - lpl(ok,i,ih,1)));
        end
        fprintf('%s', NLc);
    end
end
fprintf(['  the score column is the mean gain in that variable''s own log predictive%s' ...
    '  likelihood, so the five do not add up to the joint gain%s'], NLc, NLc);

fprintf(['%scalibration: how often the realized value fell inside the ' ...
    'interval, and how wide%s'], NLc, NLc);
show = [5 2];                            % GDP growth and PCE inflation
for ih = 1:2
    ok = ~isnan(actual(:,1,ih));
    fprintf('%s  h = %d, %d forecasts%s', NLc, hs(ih), nnz(ok), NLc);
    fprintf('  %-16s %-16s %11s %11s %11s %11s%s', 'variable', 'model', ...
        'cover 80%', 'width 80%', 'cover 95%', 'width 95%', NLc);
    for i = show
        for im = 1:nm
            u = pit(ok,i,ih,im);
            fprintf('  %-16.16s %-16s %10.0f%% %11.2f %10.0f%% %11.2f%s', ...
                vlabel(i), mname(im), 100*mean(u > .100 & u < .900), ...
                mean(w80(ok,i,ih,im)), 100*mean(u > .025 & u < .975), ...
                mean(w95(ok,i,ih,im)), NLc);
        end
    end
end
fprintf(['  a well calibrated 80%% interval covers 80%% of the time; among those that%s' ...
    '  do, the narrower one is the more useful%s'], NLc, NLc);

fprintf('%ssimulation noise: the homoskedastic model scored by simulation and exactly%s', NLc, NLc);
for ih = 1:2
    ok = ~isnan(actual(:,1,ih));
    d = lpl(ok,:,ih,1) - lpl_exact(ok,:,ih);
    fprintf('  h = %d: mean difference %.4f, largest %.4f, over %d forecasts and %d variables%s', ...
        hs(ih), mean(d(:)), max(abs(d(:))), nnz(ok), n, NLc);
end

fprintf('%sinefficiency factors at the last origin, %d draws%s', NLc, nsim, NLc);
fprintf('  VAR-CSV   phi %.0f, sigh2 %.0f, Sig(1,1) %.0f%s', IFc, NLc);
fprintf('  VAR-OISV  phi_1 %.0f, sig2_1 %.0f, h_T1 %.0f%s', IFs, NLc);

fprintf('%spredictive standard deviation of GDP growth at h = 1, median over origins%s', NLc, NLc);
for im = 1:nm
    fprintf('  %-16s targets through 2019 %6.2f, targets 2020 onwards %6.2f%s', mname(im), ...
        median(psd(tdate(:,1) < covid, 5, 1, im), 'omitnan'), ...
        median(psd(tdate(:,1) >= covid, 5, 1, im), 'omitnan'), NLc);
end

    % everything the page quotes, per origin, so a question about the split or a
    % single quarter does not need another run
save(fullfile(tdir, 'scores_by_origin.mat'), 'odate', 'tdate', 'point', 'lpl', ...
    'ljnt', 'psd', 'actual', 'lpl_exact', 'pit', 'w80', 'w95', 'mname', 'hs', ...
    'vars', 'vlabel', 'nsim', 'burnin');
fprintf('%sper-origin scores saved to scores_by_origin.mat%s', NLc, NLc);

%% ---- figures ----
figure('Position', [100 100 820 520]);
ptitle = ["all targets" "targets through 2019"];
for ib = 1:2
    for ih = 1:2
        subplot(2, 2, (ib-1)*2 + ih);
        hold on
        for im = 2:nm
            ok = ~isnan(ljnt(:,ih,im));
            if ib == 2, ok = ok & tdate(:,ih) < covid; end
            plot(odate(ok), cumsum(ljnt(ok,ih,im) - ljnt(ok,ih,1)), 'LineWidth', 1.1);
        end
        yline(0, 'k:'); hold off; box off
        title(sprintf('h = %d, %s', hs(ih), ptitle(ib)));
        if ib == 1 && ih == 1
            legend(mname(2:nm), 'Location', 'northwest', 'Box', 'off');
            ylabel('cumulative log score difference');
        end
        if ib == 2 && ih == 1, ylabel('cumulative log score difference'); end
    end
end
exportgraphics(gcf, fullfile(tdir, 'fig_cumscore.png'), 'Resolution', 150);

figure('Position', [100 100 760 320]);
hold on
for im = 1:nm
    plot(odate, psd(:,5,1,im), 'LineWidth', 1.1);
end
hold off; box off
legend(mname, 'Location', 'northwest', 'Box', 'off');
ylabel('predictive standard deviation');
title('One-quarter-ahead predictive standard deviation of GDP growth');
exportgraphics(gcf, fullfile(tdir, 'fig_psd.png'), 'Resolution', 150);

fprintf('\nbuild finished in %.1f minutes\n', toc(t_all)/60);
diary off

%% -------------------------------------------------------------------------
function [yh, ld, lj, sd] = run_draws(spec, Acol, Scol, D, cfg, n, k, H)
% One simulated path per draw, and the four quantities averaged or collected.
nd = size(Acol, 1);
yh = zeros(nd, n, H);  ld = zeros(nd, n, H);  lj = zeros(nd, H);  sd = zeros(nd, n, H);
for d = 1:nd
    dr.A = reshape(Acol(d,:), k, n);
    switch spec
        case 'gauss'
            dr.Sig = reshape(Scol(d,:,:), n, n);
        case 'csv'
            dr.Sig = reshape(Scol(d,:), n, n);
            dr.h_T = D.h(d,end);  dr.phi = D.phi(d);  dr.sigh2 = D.sigh2(d);
        case 'oisv'
            dr.impact = reshape(Scol(d,:), n, n);
            dr.h_T = D.h_T(d,:);  dr.phi = D.phi(d,:);  dr.sig2 = D.sig2(d,:);
    end
    [y1, l1, j1, s1] = bvar.forecast.simulate(spec, dr, cfg);
    yh(d,:,:) = y1';  ld(d,:,:) = l1';  lj(d,:) = j1';  sd(d,:,:) = s1';
end
end

function [point, lpl, ljnt, psd, pit, w80, w95] = store(point, lpl, ljnt, psd, ...
    pit, w80, w95, io, im, yh, ld, lj, sd, hs, nobs, t, yobs)
% Average the draws into the point forecast, the log predictive likelihoods, the
% predictive standard deviation, the predictive cdf at the realized value and
% the widths of two intervals, at the two horizons that are evaluated.
%
% The predictive distribution is the mixture over draws of N(yh, sd^2). Its cdf is
% the average of the component cdfs, which is exact and costs one line. Its
% quantiles would need a root find per interval end, 17,000 of them over this
% exercise, so the widths come from the mixture sampled once per draw, which is
% accurate enough for a width and thousands of times cheaper. The forecasts of
% forecast_now.m, where there are a few dozen quantiles rather than thousands, use
% bvar.forecast.mixquantile and no sampling.
nd = size(yh, 1);
for ih = 1:2
    h = hs(ih);
    if t + h > nobs, continue; end
    point(io,:,ih,im) = mean(yh(:,:,h), 1);
    lpl(io,:,ih,im) = bvar.util.logsumexp(ld(:,:,h)) - log(nd);
    ljnt(io,ih,im) = bvar.util.logsumexp(lj(:,h)) - log(nd);
    psd(io,:,ih,im) = sqrt(mean(sd(:,:,h).^2, 1) + var(yh(:,:,h), 0, 1));
    pit(io,:,ih,im) = mean(normcdf((yobs(:,ih)' - yh(:,:,h))./sd(:,:,h)), 1);
    ys = yh(:,:,h) + sd(:,:,h).*randn(nd, size(yh,2));
    q = quantile(ys, [.100 .900 .025 .975], 1);
    w80(io,:,ih,im) = q(2,:) - q(1,:);
    w95(io,:,ih,im) = q(4,:) - q(3,:);
end
end
