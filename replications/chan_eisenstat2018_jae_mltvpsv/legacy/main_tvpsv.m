% This is the main run file for estimating the marginal likelihood and DIC
% of the time-varying VARs in Chan and Eisenstat (2018). 
%
% This code is free to use for academic purposes only, provided that the 
% paper is cited as:
%
% Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for 
% time-varying parameter VARs with stochastic volatility, Journal of 
% Applied Econometrics, 33(4), 509-532.
%
% This code comes without technical support of any kind.  It is expected to
% reproduce the results reported in the paper. Under no circumstances will
% the authors be held responsible for any use (or misuse) of this code in
% any way.

clear; clc;
% 1: TVP-SV; 2: TVP; 3: TVP-R1-SV; 4:TVP-R2-SV; 5: TVP-R3-SV; 6: VAR-SV; 
% 7: VAR; 8: RS-VAR; 9: RS-VAR-1; 10: RS-VAR-2; 
model = 6;
cp_ml = true;      % true: compute marginal likelihood
cp_dic = false;    % true: compute DIC
r = 2;             % number of regimes for regime-switching models
p = 2;             % number of lags - p = 1,2,3,4

nsims = 20000;
burnin = 5000;
M = 10000;     % number of replications in ML estimation
nchains = 10;  % number of MCMC chains in DIC estimation

% every "simstep"^th draw is used to evaluate the integrated likelihood in 
% DIC computation
simstep = 20;    
svsims = floor( nsims / simstep );

USdata = xlsread('USdata_2014Q4.xlsx','B28:E269'); % 1954Q3-2014Q4
data = USdata(:,[1 3 4]);

Y0 = data(1:4,:);  % GDP deflator growth, real GDP growth, Fed funds rate
shortY = data(5:end,:);
[T,n] = size(shortY);
Y = reshape(shortY',T*n,1);

%% compute the marginal likelihood 
if cp_ml
    switch model
        case 1 
            TVPSV;            
            [ml, ml_std] = ml_tvpsv(Y,store_Sigtheta,store_Sigh,store_h0,...
                store_theta0,prior,bigX,M);
        case 2        
            TVP;
            [ml, ml_std] = ml_tvp(Y,store_Sig,store_Sigtheta,store_theta0,...
                prior,bigX,M);
        case 3
            TVP_R1_SV;
            [ml, ml_std] = ml_tvp_r1_sv(Y,store_beta,store_Siggam,store_Sigh,...
                store_h0,store_gam0,prior,Xtilde,W,M);
        case 4
            TVP_R2_SV;
            [ml, ml_std] = ml_tvp_r2_sv(Y,store_gam,store_Sigbeta,store_Sigh,...
                store_h0,store_beta0,prior,Xtilde,W,M);
        case 5
            TVP_R3_SV;
            [ml,ml_std] = ml_tvp_r3_sv(Y,store_beta,store_gam,store_Sigmu,store_Sigh,...
                store_h0,store_mu0,prior,Z,M);
        case 6
            VAR_SV;
            [ml, ml_std] = ml_varsv(Y,store_theta,store_Sigh,store_h0,prior,bigX,M);
        case 7
            VAR;
            [ml, ml_std] = ml_var(Y,store_theta,store_Sig,prior,bigX,M);
        case 8
            VAR_RS;
            [ml,ml_std] = ml_var_rs(Y,store_theta,store_Sig,store_P,prior,bigX,M);
        case 9
            VAR_RS_R1;
            [ml,ml_std] = ml_var_rs_r1(Y,store_theta,store_Sig,store_P,prior,bigX,M);
        case 10
            VAR_RS_R2;            
            [ml,ml_std] = ml_var_rs_r2(Y,store_theta,store_Sig,store_P,prior,bigX,M);     
    end   
    disp(' ')
    fprintf('log marginal likelihood: %.1f (%.2f)\n', ml, ml_std);
    disp(' ' );
end

%% compute the DIC
if cp_dic
    store_dic = zeros(nchains,3);
    for ii=1:nchains
    switch model
        case 1 
            TVPSV;
            [dic,pD,Dbar] = dic_tvpsv(Y,store_Sigtheta,store_Sigh,store_h0,...
                store_theta0,bigX,simstep);
        case 2        
            TVP;
            [dic,pD,Dbar] = dic_tvp(Y,store_Sig,store_Sigtheta,...
                store_theta0,bigX,simstep);
        case 3
            TVP_R1_SV;
            [dic,pD,Dbar] = dic_tvp_r1_sv(Y,store_beta,store_Siggam,store_Sigh,...
                store_h0,store_gam0,Xtilde,W,simstep);
        case 4
            TVP_R2_SV;
            [dic,pD,Dbar] = dic_tvp_r2_sv(Y,store_gam,store_Sigbeta,store_Sigh,...
                store_h0,store_beta0,Xtilde,W,simstep);
        case 5
            TVP_R3_SV;
            [dic,pD,Dbar] =  dic_tvp_r3_sv(Y,store_beta,store_gam,store_Sigmu,...
                store_Sigh,store_mu0,store_h0,Z,simstep);   
        case 6
            VAR_SV;
            [dic,pD,Dbar] = dic_varsv(Y,store_theta,store_Sigh,store_h0,...
                bigX,simstep);
        case 7
            VAR;
            [dic,pD,Dbar] = dic_var(Y,store_theta,store_Sig,bigX,1);         
    end    
    store_dic(ii,:) = [dic pD Dbar];
    end
    
    dic_hat = mean(store_dic(:,1));
    dic_std = std(store_dic(:,1))/sqrt(nchains);
    pD_hat = mean(store_dic(:,2));
    pD_std = std(store_dic(:,2))/sqrt(nchains);
    
    disp(' ')
    fprintf('DIC: %.1f (%.2f)\n', dic_hat, dic_std);
    fprintf('Effective # of parameters: %.1f (%.2f)\n', pD_hat, pD_std);
end