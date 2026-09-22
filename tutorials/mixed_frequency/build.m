%% build - regenerate the figures and numbers of tutorials/mixed_frequency/README.md
%
% The model and priors are those of the first application of Chan, Poon and Zhu (2023),
% a VAR with common stochastic volatility, estimated here on monthly data: five FRED-MD
% series and two quarterly FRED-QD series, real GDP and real private fixed investment,
% from the August 2026 vintages. Part 1 builds the data and the restrictions, Part 2
% estimates the model, Part 3 reports the monthly estimates and compares monthly GDP
% with the Brave-Butters-Kelley series. The figures are written next to this file and
% everything printed goes to a log in tempdir.
%
% Usage, from anywhere:  run tutorials/mixed_frequency/build.m

tdir = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(tdir));
logf = fullfile(tempdir, 'bvar_mixed_frequency_build_log.txt');
if exist(logf, 'file'), delete(logf); end
diary(logf);
fprintf('tutorials/mixed_frequency/build.m, %s, MATLAB %s\n', ...
    char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm')), version);
t0 = tic;
run(fullfile(repo, 'setup.m'));

%% ------------------------------------------------------------------
%  Part 1. The data and the restrictions
%  ------------------------------------------------------------------
D = readtable(fullfile(tdir, 'mf_data.csv'), 'TextType', 'string');
dlev = datetime(D.date, 'InputFormat', 'yyyy-MM');
d = dlev(2:end);                                   % the months of the growth rates
vnames = ["IP" "CPI" "Unemployment" "Payrolls" "Hours" "GDP" "Investment"];
n = numel(vnames);  T = numel(d);
iq = [6 7];                                        % the quarterly series

% IP, CPI and payrolls in 100 x log changes, with a restriction across each gap
[G, Mg, zg] = bvar.util.dlog_gaps([D.INDPRO D.CPIAUCSL D.PAYEMS]);
X = [G(:,1) G(:,2) D.UNRATE(2:end) G(:,3) D.AWHMAN(2:end)/10 nan(T,2)];
% GDP and investment: 100 x log change between quarters, in the last month of the quarter
Q = [D.GDPC1 D.FPIx];
for j = 1:2
    iobs = find(~isnan(Q(:,j)));
    X(iobs(2:end) - 1, iq(j)) = 100*diff(log(Q(iobs,j)));
end
[Mq, zq, Y, dropped] = bvar.util.mm_constraint(X, ismember(1:n, iq));
assert(~any(dropped(:)), 'every quarterly value should have its five months in the sample');
% the rows of dlog_gaps act on the three series it was given; move them to columns 1, 2, 4
[r, c] = find(Mg);
r = r(:);  c = c(:);                               % find returns rows when Mg has one row
tg = ceil(c/3);  jg = c - (tg-1)*3;  map = [1; 2; 4];
Mg = sparse(r, (tg-1)*n + map(jg), 1, size(Mg,1), T*n);
M = [Mq; Mg];  z = [zq; zg];

% the Minnesota scales of the quarterly series: 9/19 of the AR(4) residual variance of the
% quarterly growth rates, the monthly variance that gives it under the aggregation
sig2 = nan(n,1);
for j = 1:2
    g = X(~isnan(X(:,iq(j))), iq(j));
    Zq = [ones(numel(g)-4,1) g(4:end-1) g(3:end-2) g(2:end-3) g(1:end-4)];
    e = g(5:end) - Zq*(Zq\g(5:end));
    sig2(iq(j)) = 9/19*mean(e.^2);
end
p = 4;
fprintf(['\ndata: %s to %s, %d months, the first %d as initial conditions; %d quarterly ' ...
    'values of GDP and of investment, %s to %s\n'], mlab(d(1)), mlab(d(end)), T, p, ...
    numel(zq)/2, qlab(d(find(~isnan(X(:,6)), 1))), qlab(d(find(~isnan(X(:,6)), 1, 'last'))));
fprintf('missing monthly values: %s\n', strjoin(compose('%s %d', vnames', sum(isnan(Y))'), ', '));
fprintf('restrictions across gaps: %d, CPI over %s-%s\n', numel(zg), ...
    mlab(d(find(isnan(Y(:,2)), 1))), mlab(d(find(isnan(Y(:,2)), 1, 'last'))));

%% ------------------------------------------------------------------
%  Part 2. Estimation
%  ------------------------------------------------------------------
nsim = 10000;  burnin = 10000;  seed = 1;  hblock = 36;
t1 = tic;
res = bvar.models.mfvar_csv(Y, p, 'M', M, 'z', z, 'sig2', sig2, 'nsim', nsim, ...
    'burnin', burnin, 'seed', seed, 'hblock', hblock, 'draws', true);
fprintf('\n%d draws after a burn-in period of %d, seed %d: %.1f minutes\n', ...
    nsim, burnin, seed, toc(t1)/60);
fprintf('share of the %d-month blocks of h accepted: %.3f\n', hblock, res.accept_rate);
fprintf('|M*y - z| over the kept draws: max %.1e\n', res.max_resid);
fprintf('posterior means: phi %.3f, sigh2 %.3f\n', res.phi_mean, res.sigh2_mean);

% inefficiency factors of the monthly values of GDP and investment, of h, phi and sigh2
im = find(isnan(reshape(Y', [], 1)));
vm = im - (ceil(im/n) - 1)*n;
L = 100;
IF_g = bvar.diag.inefficiency_factor(double(res.draws.ym(:, vm == 6)), L);
IF_i = bvar.diag.inefficiency_factor(double(res.draws.ym(:, vm == 7)), L);
IF_h = bvar.diag.inefficiency_factor(res.draws.h, L);
IF_p = bvar.diag.inefficiency_factor([res.draws.phi res.draws.sigh2], L);
fprintf('inefficiency factors, median and largest: GDP %.1f %.1f, investment %.1f %.1f, h %.1f %.1f\n', ...
    median(IF_g), max(IF_g), median(IF_i), max(IF_i), median(IF_h), max(IF_h));
fprintf('inefficiency factors of phi and sigh2: %.1f %.1f\n', IF_p);

% the same sampler with the whole volatility path as one block, briefly, for its acceptance rate
r1 = bvar.models.mfvar_csv(Y, p, 'M', M, 'z', z, 'sig2', sig2, 'nsim', 1000, ...
    'burnin', 200, 'seed', seed, 'hblock', T - p);
fprintf('share of proposals accepted when h is one block of %d months: %.3f\n', T - p, r1.accept_rate);

%% ------------------------------------------------------------------
%  Part 3. The monthly estimates
%  ------------------------------------------------------------------
gm = res.Y_mean;  qlo = res.Y_q(:,:,1);  qhi = res.Y_q(:,:,5);
fprintf('\nmonthly growth, percent: posterior mean (90%% band)\n');
fprintf('%-8s %24s %24s\n', 'month', 'GDP', 'investment');
for t = find(d >= datetime(2020,1,1) & d <= datetime(2020,12,1))'
    fprintf('%-8s %7.2f (%6.2f, %6.2f) %7.2f (%6.2f, %6.2f)\n', mlab(d(t)), ...
        gm(t,6), qlo(t,6), qhi(t,6), gm(t,7), qlo(t,7), qhi(t,7));
end
fprintf('the posterior mean path satisfies the aggregation: max |M*y - z| %.1e\n', ...
    norm(M*reshape(gm', [], 1) - z, inf));
tq = find(d == datetime(2020,6,1));
fprintf('2020Q2, 100 x log change: GDP %.2f, investment %.2f\n', X(tq,6), X(tq,7));

% the ragged edge: July 2026, with no quarterly value yet
t = T;
fprintf('\n%s, before the third-quarter release: GDP %.2f (%.2f, %.2f), investment %.2f (%.2f, %.2f)\n', ...
    mlab(d(t)), gm(t,6), qlo(t,6), qhi(t,6), gm(t,7), qlo(t,7), qhi(t,7));
% the values that were never published: CPI inflation in the two months around the missing
% October 2025 index, whose sum is known, and the October 2025 unemployment rate
tc = find(isnan(Y(:,2)));
for t = tc'
    fprintf('CPI inflation %s: %.3f (%.3f, %.3f)\n', mlab(d(t)), gm(t,2), qlo(t,2), qhi(t,2));
end
fprintf('  their sum, from the published index: %.3f\n', zg);
t = find(isnan(Y(:,3)));
fprintf('unemployment rate %s: %.2f (%.2f, %.2f); published %s %.1f and %s %.1f\n', ...
    mlab(d(t)), gm(t,3), qlo(t,3), qhi(t,3), mlab(d(t-1)), Y(t-1,3), mlab(d(t+1)), Y(t+1,3));

% the Brave-Butters-Kelley series is 12 times the monthly growth rate; divided by 12, its
% Mariano-Murasawa aggregates must give the quarterly GDP growth of mf_data.csv
B = readtable(fullfile(tdir, 'bbk_mgdp.csv'), 'TextType', 'string');
bd = datetime(B.date, 'InputFormat', 'yyyy-MM');
bbk = B.BBKMGDP/12;
tq = find(~isnan(X(:,6)) & d >= bd(1) + calmonths(4) & d <= bd(end));
agg = zeros(numel(tq), 1);
for k = 1:numel(tq)
    agg(k) = [1 2 3 2 1]/3*bbk(bd >= d(tq(k)) - calmonths(4) & bd <= d(tq(k)));
end
fprintf(['\nBrave-Butters-Kelley divided by 12, aggregated to %d quarters: largest gap to the ' ...
    'quarterly GDP growth %.4f\n'], numel(tq), max(abs(agg - X(tq,6))));
assert(max(abs(agg - X(tq,6))) < .01, 'the Brave-Butters-Kelley series is not in the units assumed');
[dc, ia, ib] = intersect(d, bd);
x = gm(ia,6);  y = bbk(ib);
not20 = year(dc) ~= 2020;
fprintf(['\nmonthly GDP growth against Brave-Butters-Kelley, %s to %s (%d months): ' ...
    'correlation %.2f, %.2f without 2020; RMS difference %.2f, %.2f without 2020\n'], ...
    mlab(dc(1)), mlab(dc(end)), numel(dc), corr(x, y), corr(x(not20), y(not20)), ...
    sqrt(mean((x - y).^2)), sqrt(mean((x(not20) - y(not20)).^2)));
fprintf('standard deviations: this model %.2f, Brave-Butters-Kelley %.2f (without 2020: %.2f, %.2f)\n', ...
    std(x), std(y), std(x(not20)), std(y(not20)));
fprintf('%-8s %8s %8s\n', 'month', 'model', 'BBK');
for k = find(dc >= datetime(2020,2,1) & dc <= datetime(2020,7,1))'
    fprintf('%-8s %8.2f %8.2f\n', mlab(dc(k)), x(k), y(k));
end

%% ------------------------------------------------------------------
%  Figures
%  ------------------------------------------------------------------
lo68 = res.Y_q(:,:,2);  hi68 = res.Y_q(:,:,4);
fig = figure('Visible', 'off', 'Position', [100 100 900 380]);
k = d >= datetime(2006,1,1);
kb = bd >= datetime(2006,1,1);
bvar.util.shaded_band(d(k), lo68(k,6), hi68(k,6));
hold on
plot(d(k), gm(k,6), 'k', 'LineWidth', 1.2);
plot(bd(kb), bbk(kb), '--', 'Color', [.85 .33 .1], 'LineWidth', 1);
hold off; box off; grid on
ylabel('percent');
legend({'68% credible band', 'posterior mean', 'Brave-Butters-Kelley'}, 'Location', 'southwest');
exportgraphics(fig, fullfile(tdir, 'fig_gdp.png'), 'Resolution', 150);
close(fig);

fig = figure('Visible', 'off', 'Position', [100 100 900 520]);
k = d >= datetime(2019,7,1) & d <= datetime(2021,6,1);
ttl = ["GDP" "Private fixed investment"];
for j = 1:2
    subplot(2,1,j);
    bvar.util.shaded_band(d(k), lo68(k,iq(j)), hi68(k,iq(j)));
    hold on
    plot(d(k), gm(k,iq(j)), 'k', 'LineWidth', 1.2);
    hold off; box off; grid on
    title(ttl(j)); ylabel('percent');
end
exportgraphics(fig, fullfile(tdir, 'fig_2020.png'), 'Resolution', 150);
close(fig);

fprintf('\nbuild finished in %.1f minutes\n', toc(t0)/60);
diary off

function s = mlab(dt)
s = sprintf('%d:%02d', year(dt), month(dt));
end

function s = qlab(dt)
s = sprintf('%dQ%d', year(dt), ceil(month(dt)/3));
end
