% bvar.structural.check_separable - test whether a set of sign and ranking
% restrictions separates every pair of shocks, as Assumption 2 of Chan, Matthes
% and Yu (2026) requires of the restrictions given to sign_assign.
%
%   [ok,bad] = bvar.structural.check_separable(S)
%   [ok,bad] = bvar.structural.check_separable(S, Rineq)
%
%   S     : n x m sign restrictions, as bvar.structural.sign_assign takes them
%   Rineq : m x n x k ranking restrictions, as sign_assign takes them; default
%           none
%   ok    : true when every pair of shocks is separated
%   bad   : the pairs that are not, one row [j l] per pair, j < l
%
% Two shocks are separated when some variable is restricted with the same sign
% under both and another with opposite signs, or when the same pair of variables
% is ranked one way for one shock and the other way for the other. Either makes
% their admissible columns disjoint, which is what lets sign_assign draw each
% shock's column independently of the others; where it fails, two shocks can
% draw the same column and sign_assign errors. Call this once before the
% rejection loop, since sign_assign runs once per candidate rotation.
%
% Called with S alone it reports the pairs the ranking restrictions are
% carrying.
%
% See:
% Chan, J.C.C., Matthes, C. and Yu, X. (2026). Large Structural VARs with
% Multiple Sign and Ranking Restrictions, Quantitative Economics, 17(3):
% 709-740, Assumption 2.

function [ok,bad] = check_separable(S,Rineq)
[n,m] = size(S);
if nargin < 2 || isempty(Rineq)
    Rineq = zeros(m,n);
end
assert(size(Rineq,1) == m && size(Rineq,2) == n, ...
    'bvar:structural:check_separable:badRineq', ...
    'Rineq must be m x n x k with m = %d shocks and n = %d variables', m, n);
k = size(Rineq,3);

bad = zeros(0,2);
for a = 1:m-1
    for b = a+1:m
        both = ~isnan(S(:,a)) & ~isnan(S(:,b));
        by_sign = any(both & S(:,a) == S(:,b)) && any(both & S(:,a) == -S(:,b));
        by_rank = false;
        for ja = 1:k
            ra = Rineq(a,:,ja);
            if ~any(ra), continue, end
            for jb = 1:k
                if isequal(sign(ra), -sign(Rineq(b,:,jb)))
                    by_rank = true;  break
                end
            end
            if by_rank, break, end
        end
        if ~(by_sign || by_rank)
            bad(end+1,:) = [a b];   %#ok<AGROW>
        end
    end
end
ok = isempty(bad);
end
