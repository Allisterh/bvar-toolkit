function test_csv_armh_block
% Chains of bvar.sv.csv_armh_block with blocks of length 1 and 2 and with the whole path as
% one block must match the means and variances of the conditional posterior of h, computed
% on a grid for T = 3, at the default c_reject and at c_reject = .2, where the
% Metropolis-Hastings step matters.
rng(20260921, 'twister');
T = 3; n = 3; rho = .9; sigh2 = .3;
s2 = [2; 8; .5];

g = linspace(-5, 6, 111)';
[h1, h2, h3] = ndgrid(g, g, g);
Hm = [h1(:) h2(:) h3(:)];
Hrho = eye(T) - rho*diag(ones(T-1,1), -1);
Q = Hrho'*diag([(1-rho^2)/sigh2; ones(T-1,1)/sigh2])*Hrho;
lp = -.5*sum((Hm*Q).*Hm, 2) - n/2*sum(Hm, 2) - .5*exp(-Hm)*s2;
w = exp(lp - max(lp));  w = w/sum(w);
m_exact = Hm'*w;
v_exact = ((Hm - m_exact').^2)'*w;

nsweep = 25000;
for cr = [3 .2]
    for L = 1:3
        h = zeros(T,1);  H = zeros(nsweep, T);  acc = 0;  nb = 0;
        for isweep = 1:nsweep
            [h, a, b] = bvar.sv.csv_armh_block(s2, rho, sigh2, h, n, 'block', L, 'c_reject', cr);
            H(isweep,:) = h';  acc = acc + a;  nb = nb + b;
        end
        if L == 3
            assert(nb == nsweep, 'csv_armh_block: block 3 at T = 3 must be the whole path');
        end
        H = H(1001:end,:);
        assert(max(abs(mean(H)' - m_exact)) < 0.03, ...
            'csv_armh_block: block %d, c_reject %g, the posterior means differ from the grid by %.3f', ...
            L, cr, max(abs(mean(H)' - m_exact)));
        assert(max(abs(var(H)'./v_exact - 1)) < 0.06, ...
            'csv_armh_block: block %d, c_reject %g, the posterior variances differ from the grid', L, cr);
        assert(acc/nb > 0.5, 'csv_armh_block: block %d accepts only %.2f of proposals', L, acc/nb);
    end
end

[~, a, b] = bvar.sv.csv_armh_block(s2, rho, sigh2, zeros(T,1), n, 'block', 1, 'ForcedAccept', true);
assert(a == b, 'csv_armh_block: ForcedAccept must accept every block');

bad = { @() bvar.sv.csv_armh_block(s2, rho, sigh2, h, n, 'block', 0), 'a zero block length'; ...
        @() bvar.sv.csv_armh_block(s2, rho, sigh2, h, n, 'c_reject', -1), 'a negative c_reject'; ...
        @() bvar.sv.csv_armh_block(s2, rho, sigh2, h, n, 'zzz', 1), 'an unknown option'};
for ib = 1:size(bad, 1)
    try
        bad{ib,1}();
        error('%s should have errored', bad{ib,2});
    catch err
        assert(strcmp(err.identifier, 'bvar:sv:csv_armh_block:badOption'), ...
            'wrong identifier for %s: %s', bad{ib,2}, err.identifier);
    end
end
fprintf('test_csv_armh_block passed\n');
end
