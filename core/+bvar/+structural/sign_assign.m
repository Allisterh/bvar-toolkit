% bvar.structural.sign_assign - accept a candidate impact matrix when every
% shock has at least one column satisfying its restrictions, then draw one such
% assignment at random. Sign and ranking restrictions are tested together.
%
%   [ok,L] = bvar.structural.sign_assign(L, S, Rineq, k)
%
%   L     : n x n candidate impact matrix, chol(Sigtilde,'lower')*Q for a
%           rotation Q from bvar.structural.qr_sign
%   S     : n x m sign restrictions, one column per shock; +1 and -1 restrict the
%           sign of that response on impact, NaN leaves it free
%   Rineq : m x n x k ranking restrictions, k per shock, each a linear
%           combination of impact responses required to be <= 0 for that shock's
%           column; an m x n matrix when k = 1. The test is <= 0, so a row of
%           zeros imposes nothing and zeros(m,n) means sign restrictions only.
%           sign_restrict tests the same quantity strictly, where a zero row
%           rejects every candidate, so pass an empty Ridx there instead
%   k     : number of ranking restrictions per shock
%   ok    : true when an admissible assignment exists
%   L     : on acceptance, column i is the shock-i column, with signs flipped
%           where that is what made the restrictions hold and the remaining n-m
%           columns randomly permuted and re-signed; unchanged when ok is false
%
% bvar.structural.sign_restrict requires column i to satisfy shock i, the
% accept-reject scheme of Rubio-Ramirez, Waggoner and Zha (2010). The labeling
% of the columns of Q is arbitrary, so this function accepts whenever every
% shock has an admissible column, and accepts far more often as a result.
% Proposition 1 of Chan, Matthes and Yu (2026) shows the target distribution is
% unchanged. Both functions stay; see the never-merge list in
% tests/variant_map.md.
%
% THE CONDITION. Each shock draws its column with no check that another shock
% has taken it, which is safe only when no column can admit two shocks. That is
% Assumption 2 of the paper, which bvar.structural.check_separable tests; call
% it once before the rejection loop. Restrictions violating it are out of scope:
% this function raises an error when it meets one, and the paper's second
% algorithm, which enumerates the admissible set, is not implemented here.
%
% THE CALLER must accept or reject the pair (A,Sigma) and Q jointly, since
% resampling Q against a fixed posterior draw targets a different distribution
% (Arias, Rubio-Ramirez, Shin and Waggoner, 2024), and must take one draw per
% accepted pair, since two assignments from the same (Sigma,Q) differ only by a
% permutation and sign flips.
%
% On acceptance the rng consumption is one unidrnd per shock, then randperm(n-m)
% and rand(n-m,1); a rejected candidate consumes nothing.
%
% See:
% Rubio-Ramirez, J.F., Waggoner, D.F. and Zha, T. (2010). Structural Vector
% Autoregressions: Theory of Identification and Algorithms for Inference,
% Review of Economic Studies, 77(2): 665-696.
% Chan, J.C.C., Matthes, C. and Yu, X. (2026). Large Structural VARs with
% Multiple Sign and Ranking Restrictions, Quantitative Economics, 17(3): 709-740.

function [ok,L] = sign_assign(L,S,Rineq,k)
[n,m] = size(S);
if nargin < 4 || isempty(k)
    k = 1;
end

satTab = zeros(m,n); % (i,j) = 1 if the j-th column of L satisfies all restrictions for the i-th shock
                     % (i,j) = -1 if the negative of j-th column of L satisfies all restrictions
for i=1:m  % check sign restrictions & row inequilities
    idx = find(S(:,i)==-1 | S(:,i)==1);
    nidx = length(idx);
    signL = sign(L(idx,:));
    for j=1:n
        if k == 1
            if (sum(signL(:,j) == S(idx,i)) == nidx) && ...
                    (sum(Rineq(i,:,:)*L(:,j) <= 0) == 1)
                satTab(i,j) = 1;
            elseif (sum(signL(:,j) == -S(idx,i)) == nidx) && ...
                    (sum(Rineq(i,:,:)*(-L(:,j)) <= 0) == 1)
                satTab(i,j) = -1;
            end
        elseif k>1
            if (sum(signL(:,j) == S(idx,i)) == nidx) && ...
                    (sum(squeeze(Rineq(i,:,:))'*L(:,j) <= 0) == k)
                satTab(i,j) = 1;
            elseif (sum(signL(:,j) == -S(idx,i)) == nidx) && ...
                    (sum(squeeze(Rineq(i,:,:))'*(-L(:,j)) <= 0) == k)
                satTab(i,j) = -1;
            end
        end
    end
end

ok = (nnz(sum(abs(satTab),2)) == m);  % admissible set is non-empty
if ok
    reorder = zeros(1,n);
    for i=1:m
        idx = find(satTab(i,:));
        draw = idx(unidrnd(length(idx)));
        reorder(i) = draw;
        if satTab(i,draw) == -1
            L(:,draw) = -L(:,draw);
        end
    end
    reorder(m+1:end) = setdiff(1:n,reorder);
    L = L(:,reorder);
        % randomly permute and switch the signs of the last n-m columns
    tmpL2 = L(:,m+1:end);
    tmpL2 = tmpL2(:,randperm(n-m))*diag(2*(rand(n-m,1)>.5) - 1);
    L(:,m+1:end) = tmpL2;
end
end
