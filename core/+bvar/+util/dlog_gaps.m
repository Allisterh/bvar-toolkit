% bvar.util.dlog_gaps - growth rates of series in levels with gaps, and the linear
% restrictions that tie the growth rates across each gap to the change over it.
%
%   [G, M, z] = bvar.util.dlog_gaps(Lv)
%
%   Lv : T x n levels, NaN where a level is missing
%   G  : (T-1) x n growth rates, G(t-1,:) = 100*(log Lv(t,:) - log Lv(t-1,:)), NaN
%        wherever either level is missing
%   M  : q x (T-1)n sparse, one row per gap with an observed level on each side, acting on
%        g = vec(G'), the stacking of bvar.util.select_obs
%   z  : q x 1, 100 times the log change over each such gap
%
% A gap of m missing levels leaves m+1 growth rates missing whose sum is known; a gap at
% either end of the sample leaves no restriction. Written for this toolkit.

function [G, M, z] = dlog_gaps(Lv)
if ~isnumeric(Lv) || ~ismatrix(Lv) || size(Lv,1) < 2
    error('bvar:util:dlog_gaps:badData', 'Lv must be a numeric matrix with at least two rows');
end
if any(Lv(:) <= 0)
    error('bvar:util:dlog_gaps:badData', 'the levels must be positive');
end
[T, n] = size(Lv);
G = 100*diff(log(Lv));
rows = []; cols = []; z = [];
q = 0;
for i = 1:n
    obs = find(~isnan(Lv(:,i)));
    for j = find(diff(obs) > 1)'
        a = obs(j);  b = obs(j+1);                  % observed levels around the gap
        q = q + 1;
        s = (a:b-1)';                                % the growth rates G(a:b-1, i)
        rows = [rows; q*ones(numel(s),1)]; %#ok<AGROW>
        cols = [cols; (s-1)*n + i]; %#ok<AGROW>
        z = [z; 100*(log(Lv(b,i)) - log(Lv(a,i)))]; %#ok<AGROW>
    end
end
M = sparse(rows, cols, 1, q, (T-1)*n);
end
