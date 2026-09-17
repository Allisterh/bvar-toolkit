%% GOLDEN-RUN VARIANT PATCH (2026-09-17): identical to legacy/main_ACP_jointden.m except
%% that the figure block at the end is replaced by a numeric capture, which prints a
%% summary of the marginal-likelihood surface and saves the surface and the
%% symmetric-prior optimum to golden_jointden.mat. The computation is untouched.
% This script reproduces the contour plot of the joint posterior density 
% of kappa1_tilde and kappa2_tilde in Chan (2022)
%
% This code is free to use for academic purposes only, provided that the 
% paper is cited as:
%
% Chan, J.C.C. (2022). Asymmetric Conjugate Priors for Large Bayesian VARs,
% Quantitative Economics, 13(3): 1145-1169
%
% This code comes without technical support of any kind. It is expected to
% reproduce the results reported in the paper. Under no circumstances will
% the authors be held responsible for any use (or misuse) of this code in
% any way.

clear; clc;
p = 5;          % if p > 8, need to change Y0 and Y below
addpath('./utility');
    % load data
data = xlsread('database_2019Q4.xlsx');
idx_ns = [1,2,4,5,10,11,12,13,15]; % index for variables in levels
Y0 = data(1:8,1:15);  % save the first 8 obs as the initial conditions
Y = data(9:end,1:15);
[T,n] = size(Y);
tmpY = [Y0(end-p+1:end,:); Y];
Z = zeros(T,n*p); 
for ii=1:p
    Z(:,(ii-1)*n+1:ii*n) = tmpY(p-ii+1:end-ii,:);
end
Z = [ones(T,1) Z];
sig2 = get_resid_var(Y0,Y);
kappa = [.04,.04^2,1,100];

    % plot ml concour
[Kappa1,Kappa2] = meshgrid(0.01:.001:.2,.001:.0002:0.012);
store_lml = zeros(size(Kappa1,1),size(Kappa2,2));
for ii = 1:size(Kappa1,1)
    for ij = 1:size(Kappa2,2)
         store_lml(ii,ij) = ml_VAR_ACP(p,Y,Z,...
             prior_ACP_redu(n,p,[Kappa1(ii,ij),Kappa2(ii,ij),kappa(3),kappa(4)],sig2,idx_ns));
    end    
end
store_ml = exp(store_lml-max(max(store_lml)));
[ml_Sym,kappa_Sym] = get_OptSymKappa(Y0,Y,Z,p,'redu',idx_ns);
% --- GOLDEN-RUN ADDITION (2026-09-17): numeric capture in place of the figures ---
[lml_max,imax] = max(store_lml(:));
[~,i1] = min(abs(Kappa1(1,:)-.04));
[~,i2] = min(abs(Kappa2(:,1)-.0016));
fprintf('grid: %d kappa2 values x %d kappa1 values\n', size(store_lml,1), size(store_lml,2));
fprintf('max log-ML on the grid: %.10g at kappa1 = %.4g, kappa2 = %.5g\n', lml_max, Kappa1(imax), Kappa2(imax));
fprintf('log-ML at the subjective-prior point (%.4g, %.5g): %.10g\n', Kappa1(i2,i1), Kappa2(i2,i1), store_lml(i2,i1));
fprintf('symmetric-prior optimum: kappa1 = kappa2 = %.10g, log-ML = %.10g\n', kappa_Sym(1), ml_Sym);
fprintf('sum of log-ML over the grid: %.17g\n', sum(store_lml(:)));
save('golden_jointden.mat','store_lml','ml_Sym','kappa_Sym');
