% bvar.util.kron_gram - weighted Gram matrix of kron(B0,X), formed from its k x k
% blocks. With Xtilde = kron(B0,X).*sqrt(vec(W)) and ytilde = vec(U).*sqrt(vec(W)),
% it returns Xtilde'*Xtilde, Xtilde'*ytilde and ytilde'*ytilde without forming
% Xtilde, in O(nTk^2 + n^3k^2) operations against O(Tn^3k^2).
%
%   [XWX, XWy, yWy] = bvar.util.kron_gram(X, B0, W, U)
%
%   X   : T x k regressors
%   B0  : n x n matrix
%   W   : T x n weights, W(t,i) the inverse variance of equation i at time t
%   U   : T x n, typically Y*B0'
%   XWX : nk x nk; block (j,l) is the sum over i of B0(i,j)*B0(i,l)*X'*diag(W(:,i))*X
%   XWy : nk x 1, vec(X'*(U.*W)*B0)
%   yWy : sum(sum(U.^2.*W))

function [XWX, XWy, yWy] = kron_gram(X, B0, W, U)
k = size(X, 2);
n = size(B0, 1);
G = zeros(k*k, n);
for i = 1:n
    G(:,i) = reshape(X'*(X.*W(:,i)), k*k, 1);      % X'*diag(W(:,i))*X
end
Cf = zeros(n, n*n);                                % Cf(i,(j-1)*n+l) = B0(i,j)*B0(i,l)
for j = 1:n
    Cf(:,(j-1)*n+(1:n)) = B0(:,j).*B0;
end
XWX = reshape(permute(reshape(G*Cf, k, k, n, n), [1 4 2 3]), n*k, n*k);
XWy = reshape(X'*(U.*W)*B0, n*k, 1);
yWy = sum(sum(U.^2.*W));
end
