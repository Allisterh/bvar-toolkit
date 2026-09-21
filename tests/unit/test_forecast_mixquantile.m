function test_forecast_mixquantile
% bvar.forecast.mixquantile must return the normal quantile when every component
% is the same, must invert the mixture cdf in general, must be monotone in p, and
% must differ from the single normal fitted to the mixture's first two moments
% whenever the components disagree.
rng(20260922, 'twister');
p = [.05 .1 .25 .5 .75 .9 .95];

%% ---- one component repeated is a normal ----
mu = 1.5*ones(50,1);  sd = 0.8*ones(50,1);
q = bvar.forecast.mixquantile(mu, sd, p);
assert(max(abs(q - norminv(p, 1.5, 0.8))) < 1e-9, ...
    'identical components must give the normal quantile');

%% ---- the general case inverts the cdf ----
D = 400;
mu = randn(D,1) + 0.3*(rand(D,1) > .7);         % a lump of shifted components
sd = 0.4 + 0.6*rand(D,1);
q = bvar.forecast.mixquantile(mu, sd, p);
F = arrayfun(@(x) mean(normcdf((x - mu)./sd)), q);
assert(max(abs(F - p)) < 1e-9, 'the quantiles do not invert the mixture cdf');
assert(all(diff(q) > 0), 'the quantiles must increase with p');

%% ---- the shape matches the shape of p ----
P = [.1 .9; .25 .75];
Q = bvar.forecast.mixquantile(mu, sd, P);
assert(isequal(size(Q), size(P)), 'the output must have the shape of p');
assert(Q(1,1) < Q(2,1) && Q(2,2) < Q(1,2), 'the quantiles are misplaced');

%% ---- a skewed mixture is not the normal fitted to its moments ----
mu = [zeros(90,1); 6*ones(10,1)];               % ten components far to the right
sd = ones(100,1);
q = bvar.forecast.mixquantile(mu, sd, [.5 .95]);
m = mean(mu);  v = mean(sd.^2) + var(mu, 1);    % the mixture's first two moments
approx = norminv([.5 .95], m, sqrt(v));
assert(abs(q(1) - approx(1)) > .3 && abs(q(2) - approx(2)) > .3, ...
    'a skewed mixture should not match the normal fitted to its moments');
assert(q(1) < m, 'the median of this mixture sits below its mean');

%% ---- argument checks ----
bad = { @() bvar.forecast.mixquantile(mu, sd(1:end-1), .5), 'badPair'; ...
        @() bvar.forecast.mixquantile(mu, -sd, .5),         'badSd'; ...
        @() bvar.forecast.mixquantile(mu, sd, 0),           'badProb'; ...
        @() bvar.forecast.mixquantile(mu, sd, 1),           'badProb'};
for ib = 1:size(bad, 1)
    try
        bad{ib,1}();
        error('call %d should have errored (%s)', ib, bad{ib,2});
    catch err
        assert(strcmp(err.identifier, ['bvar:forecast:mixquantile:' bad{ib,2}]), ...
            'call %d gave %s, expected %s', ib, err.identifier, bad{ib,2});
    end
end
end
