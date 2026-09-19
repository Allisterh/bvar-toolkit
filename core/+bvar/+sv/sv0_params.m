% bvar.sv.sv0_params - posterior draw of the SV state-equation parameters
% (phi, sig2) for stationary zero-mean AR(1) log-volatilities.
%
%   [phi,sig2,flag_phi] = bvar.sv.sv0_params(h, phi, Hyper)
%   [phi,sig2,flag_phi] = bvar.sv.sv0_params(h, phi, Hyper, phi_bnd)
%   [phi,sig2,flag_phi] = bvar.sv.sv0_params(h, phi, Hyper, phi_bnd, 'proposal', 'truncated')
%
%   h        : T x n zero-mean log-volatility paths, one column per series
%   phi      : n x 1 current AR(1) coefficients; the new draw on output
%   Hyper    : struct with nuh, Sh (inverse-gamma prior on sig2) and phi0,
%              Vphi (normal prior on phi)
%   phi_bnd  : truncation bound on the phi candidate, which can be accepted
%              only if |phic| < phi_bnd; default .99, the OISV canonical value
%   'proposal' : the Metropolis-Hastings candidate for phi. 'untruncated'
%              (default) draws it from the normal part of its conditional,
%              N(phi_hat, 1/Kphi), and rejects it unless |phic| < phi_bnd;
%              'truncated' draws it from that normal truncated to
%              (-phi_bnd, phi_bnd). Both target the same conditional.
%   sig2     : n x 1 innovation variances
%   flag_phi : n x 1, 1 where the phi candidate was accepted
%
% NEVER merge with bvar.sv.sv_params. This is not that sampler at mu = 0: the
% OISV pair keeps the zero-mean sampler separate, with a different truncation
% bound (.99 here, .999 there).
%
% See:
% Chan, J.C.C., Koop, G. and Yu, X. (2024). Large Order-Invariant Bayesian
% VARs with Stochastic Volatility, Journal of Business and Economic
% Statistics, 42(2): 825-837.

function [phi,sig2,flag_phi] = sv0_params(h,phi,Hyper,phi_bnd,varargin)
if nargin < 4 || isempty(phi_bnd)
    phi_bnd = .99;      % OISV canonical truncation bound
end
proposal = 'untruncated';
if mod(numel(varargin), 2) ~= 0
    error('bvar:sv:sv0_params:badOption', 'options must come in name-value pairs');
end
for iv = 1:2:numel(varargin)
    switch lower(char(string(varargin{iv})))
        case 'proposal', proposal = lower(char(string(varargin{iv+1})));
        otherwise, error('bvar:sv:sv0_params:badOption', 'unknown option ''%s''', ...
                char(string(varargin{iv})));
    end
end
if ~any(strcmp(proposal, {'untruncated', 'truncated'}))
    error('bvar:sv:sv0_params:badOption', ...
        'proposal must be ''untruncated'' or ''truncated''; got ''%s''', proposal);
end
[T,n] = size(h);

    % sample sig2
e_h = [h(1,:).*sqrt(1-phi.^2)'; h(2:end,:)-repmat(phi',T-1,1).*h(1:end-1,:)];
sig2 = 1./gamrnd(Hyper.nuh+T/2,1./(Hyper.Sh + sum(e_h.^2)'/2));

    % sample phi: its conditional is N(phi_hat, 1/Kphi) times exp(g_phi), the part
    % of the density of h(1) that depends on phi, on |phi| < phi_bnd
Kphi = 1./Hyper.Vphi + sum(h(1:T-1,:).^2)'./sig2;
phi_hat = (Hyper.phi0./Hyper.Vphi + sum(h(1:T-1,:).*h(2:T,:))'./sig2)./Kphi;
flag_phi = zeros(n,1);
if strcmp(proposal, 'untruncated')
    phic = phi_hat + 1./sqrt(Kphi).*randn(n,1);
    for ii = 1:n
        g_phi = @(x) .5*log(1-x^2) -.5*(1-x^2)/sig2(ii)*h(1,ii)^2;
        if abs(phic(ii))<phi_bnd
            alpMH = exp(g_phi(phic(ii))-g_phi(phi(ii)));
            if alpMH>rand
                phi(ii) = phic(ii);
                flag_phi(ii) = 1;
            end
        end
    end
else
    for ii = 1:n
        g_phi = @(x) .5*log(1-x^2) -.5*(1-x^2)/sig2(ii)*h(1,ii)^2;
        phic = tnorm_draw(phi_hat(ii), 1/sqrt(Kphi(ii)), -phi_bnd, phi_bnd);
        if exp(g_phi(phic)-g_phi(phi(ii))) > rand
            phi(ii) = phic;
            flag_phi(ii) = 1;
        end
    end
end
end

function x = tnorm_draw(mu, s, a, b)
% One draw from N(mu, s^2) truncated to (a, b), by the inverse transform on the
% side of the interval nearest the mean, so an interval deep in a tail keeps its
% precision; beyond about 37 standard deviations, where the normal cdf
% underflows, by the exponential rejection sampler of Robert, C.P. (1995),
% Simulation of Truncated Normal Variables, Statistics and Computing, 5(2):
% 121-125.
alpha = (a - mu)/s;
beta = (b - mu)/s;
if beta < 0                                    % the interval lies below the mean
    Fa = normcdf(alpha);
    Fb = normcdf(beta);
    if Fb > 0
        z = norminv(Fa + rand*(Fb - Fa));
    else
        z = -tail_draw(-beta, -alpha);
    end
elseif alpha > 0                               % the interval lies above the mean
    Ga = normcdf(alpha, 'upper');
    Gb = normcdf(beta, 'upper');
    if Ga > 0
        z = -norminv(Gb + rand*(Ga - Gb));
    else
        z = tail_draw(alpha, beta);
    end
else
    Fa = normcdf(alpha);
    Fb = normcdf(beta);
    z = norminv(Fa + rand*(Fb - Fa));
end
x = mu + s*z;
end

function z = tail_draw(c, d)
% N(0,1) truncated to (c, d) with c far in the upper tail
lam = (c + sqrt(c^2 + 4))/2;
while true
    z = c - log(rand)/lam;
    if z < d && rand <= exp(-(z - lam)^2/2)
        return
    end
end
end
