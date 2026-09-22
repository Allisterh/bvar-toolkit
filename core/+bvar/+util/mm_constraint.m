% bvar.util.mm_constraint - the linear restrictions M*y = z that tie quarterly growth
% rates to the monthly growth rates of the same series, by the log-linear aggregation of
% Mariano and Murasawa (2003).
%
%   [M, z, Y] = bvar.util.mm_constraint(X, isq)
%   [M, z, Y, dropped] = bvar.util.mm_constraint(X, isq, 'weights', w)
%
%   X       : T x n monthly data. A quarterly column holds the quarterly growth rate, 100
%             times the log change, in the last month of each quarter and NaN elsewhere
%   isq     : logical n-vector, true for the quarterly columns
%   'weights' : w(j) multiplies the monthly value j-1 months before the quarterly one;
%             default [1 2 3 2 1]/3, so z_t = (y_t + 2y_{t-1} + 3y_{t-2} + 2y_{t-3} + y_{t-4})/3
%   M       : q x Tn sparse, one row per quarterly value, acting on y = vec(Y'), the
%             stacking of bvar.util.select_obs
%   z       : q x 1 quarterly values
%   Y       : X with every entry of the quarterly columns set to NaN, the monthly
%             values to be drawn
%   dropped : T x n logical, the quarterly values left out because their window starts
%             before the first month
%
% The monthly unknowns are monthly growth rates in percent per month. The monthly
% columns pass through unchanged, NaN included. Written for this toolkit.
%
% See:
% Mariano, R.S. and Murasawa, Y. (2003). A New Coincident Index of Business Cycles
% Based on Monthly and Quarterly Series, Journal of Applied Econometrics, 18(4):
% 427-443.
% Chan, J.C.C., Poon, A. and Zhu, D. (2023). High-Dimensional Conditionally Gaussian
% State Space Models with Missing Data, Journal of Econometrics, 236(1): 105468,
% Section 2.2.

function [M, z, Y, dropped] = mm_constraint(X, isq, varargin)
w = [1 2 3 2 1]/3;
if mod(numel(varargin), 2) ~= 0
    error('bvar:util:mm_constraint:badOption', 'options must come in name-value pairs');
end
for iv = 1:2:numel(varargin)
    switch lower(char(string(varargin{iv})))
        case 'weights', w = varargin{iv+1}(:)';
        otherwise, error('bvar:util:mm_constraint:badOption', 'unknown option ''%s''', ...
                char(string(varargin{iv})));
    end
end
if ~isnumeric(X) || ~ismatrix(X) || isempty(X)
    error('bvar:util:mm_constraint:badX', 'X must be a nonempty T x n numeric matrix');
end
[T, n] = size(X);
isq = logical(isq(:)');
if numel(isq) ~= n || ~any(isq)
    error('bvar:util:mm_constraint:badIsq', 'isq must be a logical %d-vector with at least one true entry', n);
end
if isempty(w) || ~all(isfinite(w))
    error('bvar:util:mm_constraint:badOption', 'the weights must be a nonempty finite vector');
end
L = numel(w);

rows = []; cols = []; vals = []; z = [];
dropped = false(T, n);
q = 0;
for i = find(isq)
    for t = find(~isnan(X(:,i)))'
        if t < L
            dropped(t,i) = true;
            continue
        end
        q = q + 1;
        s = t - (0:L-1);                           % months t, t-1, ..., t-L+1
        rows = [rows; q*ones(L,1)]; %#ok<AGROW>
        cols = [cols; (s(:)-1)*n + i]; %#ok<AGROW>
        vals = [vals; w(:)]; %#ok<AGROW>
        z = [z; X(t,i)]; %#ok<AGROW>
    end
end
M = sparse(rows, cols, vals, q, T*n);
Y = X;
Y(:, isq) = NaN;
end
