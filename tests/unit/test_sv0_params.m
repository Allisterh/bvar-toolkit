function test_sv0_params
% Seeded draw-for-draw equivalence with the legacy OISV sample_SV0para under
% defaults. With 'proposal' set to 'truncated', the step must leave the posterior of
% phi given h invariant, for a path whose posterior sits against the bound, one
% with an interior mode and one against the negative bound, and its truncated
% normal draw must stay inside the bound when the normal is centered tens of
% thousands of standard deviations beyond it.
root = getappdata(0, 'bvar_repo_root');
leg = fullfile(root, 'replications', 'chan_koop_yu2024_jbes_oisv', 'legacy', 'utility');
addpath(leg); c = onCleanup(@() rmpath(leg));

T = 70; n = 4;
rng(51, 'twister');
h = 0.5*randn(T, n);
phi_in = 0.9*ones(n, 1);
Hyper = struct('nuh', 3*ones(n,1), 'Sh', 0.1*ones(n,1), ...
    'phi0', 0.95*ones(n,1), 'Vphi', 0.05^2*ones(n,1));
rng(52, 'twister');
[phi1, sig21, f1] = bvar.sv.sv0_params(h, phi_in, Hyper);
rng(52, 'twister');
[phi2, sig22, f2] = sample_SV0para(h, phi_in, Hyper);
assert(isequal(phi2,phi1) && isequal(sig22,sig21) && isequal(f2,f1), ...
    'sv0_params: differs from legacy sample_SV0para');
assert(any(f1 == 1), 'sv0_params: degenerate test, no phi candidate ever accepted');

% explicit default bound gives the identical draw
rng(52, 'twister');
[phi3, sig23, f3] = bvar.sv.sv0_params(h, phi_in, Hyper, .99);
assert(isequal(phi3,phi1) && isequal(sig23,sig21) && isequal(f3,f1), ...
    'sv0_params: explicit phi_bnd=.99 differs from default');
rng(52, 'twister');
[phi4, sig24, f4] = bvar.sv.sv0_params(h, phi_in, Hyper, [], 'proposal', 'untruncated');
assert(isequal(phi4,phi1) && isequal(sig24,sig21) && isequal(f4,f1), ...
    'sv0_params: ''untruncated'' must be the default');

    % 'truncated' against the posterior of phi given h, with sig2 integrated out:
    %   p(phi | h) propto N(phi; phi0, Vphi) (1-phi^2)^(1/2) (Sh + SSR(phi)/2)^-(nuh+T/2)
    % on |phi| < .99, where SSR(phi) = (1-phi^2) h_1^2 + sum_t (h_t - phi h_{t-1})^2
H1 = struct('nuh', 3, 'Sh', .1, 'phi0', .95, 'Vphi', .05^2);
paths = [.999 .02 400; .8 .1 200; -.999 .02 400];      % phi, sig2 and T of each path
grid = linspace(-.99, .99, 400001)'; grid = grid(2:end-1);
for ic = 1:size(paths, 1)
    rng(60 + ic, 'twister');
    T1 = paths(ic,3); x = zeros(T1, 1);
    x(1) = sqrt(paths(ic,2)/(1 - paths(ic,1)^2))*randn;
    for t = 2:T1, x(t) = paths(ic,1)*x(t-1) + sqrt(paths(ic,2))*randn; end
    SSR = (1 - grid.^2)*x(1)^2 + sum(x(2:end).^2) - 2*grid*sum(x(2:end).*x(1:end-1)) ...
        + grid.^2*sum(x(1:end-1).^2);
    lp = -.5*(grid - H1.phi0).^2/H1.Vphi + .5*log(1 - grid.^2) - (H1.nuh + T1/2)*log(H1.Sh + SSR/2);
    w = exp(lp - max(lp)); w = w/sum(w);
    m = sum(w.*grid); s = sqrt(sum(w.*(grid - m).^2));
    M = 20000; d = zeros(M, 1); ph = .5;
    rng(70 + ic, 'twister');
    for r = 1:M
        ph = bvar.sv.sv0_params(x, ph, H1, [], 'proposal', 'truncated');
        d(r) = ph;
    end
    d = d(1001:end);
    assert(abs(mean(d) - m) < 4*bvar.diag.mcse(d, 100) && abs(std(d)/s - 1) < .1, ...
        'sv0_params(truncated), path %d: mean %.5f sd %.5f, posterior %.5f and %.5f', ...
        ic, mean(d), std(d), m, s);
end

    % a prior centered at 1.5 or -1.5 with a tiny variance puts the normal tens of
    % thousands of standard deviations beyond the bound; the draw stays inside it
for c = [1.5 -1.5]
    H2 = struct('nuh', 3, 'Sh', .1, 'phi0', c, 'Vphi', 1e-10);
    ph = sign(c)*.98; moved = false;
    rng(80, 'twister');
    for r = 1:50
        [ph, ~, f] = bvar.sv.sv0_params(x, ph, H2, [], 'proposal', 'truncated');
        if f, moved = true; break, end
    end
    assert(moved && abs(ph) < .99 && abs(ph) > .99 - 1e-6 && sign(ph) == sign(c), ...
        'sv0_params(truncated): tail draw %.10f for a prior centered at %.1f', ph, c);
end

expect_error(@() bvar.sv.sv0_params(h, phi_in, Hyper, [], 'proposal', 'nw'), ...
    'bvar:sv:sv0_params:badOption');
expect_error(@() bvar.sv.sv0_params(h, phi_in, Hyper, [], 'proposal'), ...
    'bvar:sv:sv0_params:badOption');
end

function expect_error(f, id)
try
    f();
catch err
    assert(strcmp(err.identifier, id), 'expected %s, got %s: %s', id, err.identifier, err.message);
    return
end
error('expected error %s, none thrown', id);
end
