function one_factor_patch(file, rel)
% one_factor_patch - apply the one-factor substitutions to a copy of a legacy file.
%
%   one_factor_patch(file, rel)
%
% The library solves a linear system in K with the Cholesky factor C of K that
% it also uses for the draw, as (C')\(C\b). The legacy code solves it as K\b,
% which factors K a second time, and the two solutions differ in the last bits.
% Tests that compare the library with legacy code bitwise apply these
% substitutions to their copy FILE of the legacy file REL, the path under
% replications/ with forward slashes. Each substituted string must occur
% exactly once. Only the solves change; a legacy draw line or log determinant
% that factors K again obtains the same factor.

newton = {'newht = Kh\(fh+Gh.*ht);', ...
    'CKh = chol(Kh,''lower''); newht = (CKh'')\(CKh\(fh+Gh.*ht));'};
svrw_upper = {'hhat = Ph\(Kh*alph + invOmega*(Ystar-dconst));', ...
    'hhat = Ch\(Ch''\(Kh*alph + invOmega*(Ystar-dconst)));'};
sample_sv = {'h_hat = Kh\(mu/sig2*HiSH*ones(T,1) + iOmega*(ystar-d));', ...
    'CKh = chol(Kh,''lower''); h_hat = (CKh'')\(CKh\(mu/sig2*HiSH*ones(T,1) + iOmega*(ystar-d)));'};
kron_A = @(rhs) {['Ahat = KA\(sparse(1:k,1:k,VA0)\A0 + ' rhs ');'], ...
    ['CKA = chol(KA,''lower''); Ahat = (CKA'')\(CKA\(sparse(1:k,1:k,VA0)\A0 + ' rhs '));']};

subs = {
    % auxiliary mixture samplers: bvar.sv.ksc_rw_h0, ksc_rw_diffuse, ksc_ar1_mean
    'chan2020_springer_largebvar/legacy/SVRW.m',                 svrw_upper
    'chan2020_jbes_kronecker/legacy/realtime_forecasts/SVRW.m',  svrw_upper
    'chan2021_ijf_mahp/legacy/SVRW.m',                           svrw_upper
    'chan2023_jbes_hybtvp/legacy/utility/sample_SVRW.m', ...
        {'hhat = Kh\(HiSH*muh + iOmega*(ystar-d));', ...
         'CKh = chol(Kh,''lower''); hhat = (CKh'')\(CKh\(HiSH*muh + iOmega*(ystar-d)));'}
    'chan_eisenstat2018_jae_mltvpsv/legacy/SVRW.m', ...
        {'h_hat = Kh\(HiSH_h*alph + iOmega*(Ystar-dconst));', ...
         'CKh = chol(Kh,''lower''); h_hat = (CKh'')\(CKh\(HiSH_h*alph + iOmega*(Ystar-dconst)));'}
    'chan_jeliazkov2009_statespace/legacy/sp_code/SVRW.m', ...
        {'hhat = Kh\(invSigystar*(ystar-d));', 'hhat = (Ch'')\(Ch\(invSigystar*(ystar-d)));'}
    'chan2023_joe_mlvarsv/legacy/utility/sample_SV.m',           sample_sv
    'chan_koop_yu2024_jbes_oisv/legacy/utility/sample_SV.m',     sample_sv
    % the Newton-Raphson step of the common-volatility mode search:
    % bvar.sv.csv_armh, bvar.ml.intlike_csv, intlike_csv_ma
    'chan2023_joe_mlvarsv/legacy/utility/sample_CSV.m',            newton
    'chan2020_jbes_kronecker/legacy/sample_h.m',                   newton
    'chan2020_jbes_kronecker/legacy/realtime_forecasts/sample_h.m', newton
    'chan2020_springer_largebvar/legacy/sample_h.m',               newton
    'chan2020_jbes_kronecker/legacy/ml_BVAR_CSV.m',                newton
    'chan2020_jbes_kronecker/legacy/intlike_BVAR_CSV.m',           newton
    'chan2020_jbes_kronecker/legacy/intlike_BVAR_CSV_MA.m',        newton
    % bvar.ml.lniwpdf
    'chan2020_jbes_kronecker/legacy/lniwpdf.m', ...
        {'trace(Sig\(S0+tmp''*iVA0*tmp))', 'trace((CSig'')\(CSig\(S0+tmp''*iVA0*tmp)))'}
    % bvar.structural.b0_row_sampler and bvar.samplers.alp_tri_cs
    'chan_koop_yu2024_jbes_oisv/legacy/SVARSV_MH.m', ...
        {'mui = Kbi\(Hyper.B0(ii,:)./Hyper.VB0(ii,:))'';', ...
         'CKbi = chol(Kbi,''lower''); mui = (CKbi'')\(CKbi\(Hyper.B0(ii,:)./Hyper.VB0(ii,:))'');'}
    'chan_koop_yu2024_jbes_oisv/legacy/CS_MH.m', ...
        {'alpi_hat = Kalpi\(X_alpi''*iD*E(:,ii));', ...
         'CKalpi = chol(Kalpi,''lower''); alpi_hat = (CKalpi'')\(CKalpi\(X_alpi''*iD*E(:,ii)));'}
    'chan2023_joe_mlvarsv/legacy/VAR_ARSV_redu.m', ...
        {'betai_hat = Kbetai\(X_betai''*iD*E(:,ii));', ...
         'CKbetai = chol(Kbetai,''lower''); betai_hat = (CKbetai'')\(CKbetai\(X_betai''*iD*E(:,ii)));'}
    'chan2023_joe_mlvarsv/legacy/VAR_ARSVO_redu.m', ...
        {'betai_hat = Kbetai\(X_betai''*iD*E(:,ii));', ...
         'CKbetai = chol(Kbetai,''lower''); betai_hat = (CKbetai'')\(CKbetai\(X_betai''*iD*E(:,ii)));'}
    % bvar.samplers.factor_fsv
    'chan2023_joe_mlvarsv/legacy/VAR_FSV.m', ...
        {'f_hat = Kf\(XfiSig*e);', 'CKf = chol(Kf,''lower''); f_hat = (CKf'')\(CKf\(XfiSig*e));'}
    % the ml_varsv marginal likelihoods: bvar.ml.mlvarsv_csv, mlvarsv_fsv
    'chan2023_joe_mlvarsv/legacy/utility/ml_var_csv.m', ...
        {'A_hat = K_A\(sparse(1:k,1:k,Hyper.VA)\Hyper.A0 + XiOh*Y);', ...
         'CK_A = chol(K_A,''lower''); A_hat = (CK_A'')\(CK_A\(sparse(1:k,1:k,Hyper.VA)\Hyper.A0 + XiOh*Y));'}
    'chan2023_joe_mlvarsv/legacy/utility/ml_var_fsv.m', ...
        {'XiSy = bigX''/Sy;', 'CSy = chol(Sy,''lower''); XiSy = ((CSy'')\(CSy\bigX))'';'; ...
         'y''*(Sy\y)', 'y''*((CSy'')\(CSy\y))'}
    % the VAR coefficients and the error covariance in the run_all.m drivers
    'chan2023_joe_mlvarsv/legacy/VAR_NCP.m', ...
        {'A_hat = K_A\(sparse(1:k,1:k,Hyper.VA)\Hyper.A0 + XX*A_tilde);', ...
         'CK_A = chol(K_A,''lower''); A_hat = (CK_A'')\(CK_A\(sparse(1:k,1:k,Hyper.VA)\Hyper.A0 + XX*A_tilde));'}
    'chan2023_joe_mlvarsv/legacy/VAR_CSV.m', ...
        {'A_hat = K_A\(sparse(1:k,1:k,Hyper.VA)\Hyper.A0 + XiOh*Y);', ...
         'CK_A = chol(K_A,''lower''); A_hat = (CK_A'')\(CK_A\(sparse(1:k,1:k,Hyper.VA)\Hyper.A0 + XiOh*Y));'; ...
         'Q = diag((A-Hyper.A0)*(Sig\(A-Hyper.A0)''));', ...
         'Q = diag((A-Hyper.A0)*((CSig'')\(CSig\(A-Hyper.A0)'')));'}
    'chan2023_jbes_hybtvp/legacy/main_HYB_TVPSV.m', ...
        {'mui_hat = Kmui\(WiSig*Yi);', ...
         'CKmui = chol(Kmui,''lower''); mui_hat = (CKmui'')\(CKmui\(WiSig*Yi));'}
    'chan2020_jbes_kronecker/legacy/BVAR_t.m',        kron_A('XiOm*shortY')
    'chan2020_jbes_kronecker/legacy/BVAR_CSV.m',      kron_A('XiOh*shortY')
    'chan2020_jbes_kronecker/legacy/BVAR_MA.m',       kron_A('XtldiO*Ytld')
    'chan2020_jbes_kronecker/legacy/BVAR_t_CSV.m',    kron_A('XiOm*shortY')
    'chan2020_jbes_kronecker/legacy/BVAR_t_MA.m',     kron_A('XiO*Ytld')
    'chan2020_jbes_kronecker/legacy/BVAR_CSV_MA.m',   kron_A('XiO*Ytld')
    'chan2020_jbes_kronecker/legacy/BVAR_CSV_t_MA.m', kron_A('XiO*Ytld')
    };

row = find(strcmp(subs(:,1), rel));
assert(isscalar(row), 'one_factor_patch: no substitutions declared for %s', rel);
pairs = subs{row,2};
fid = fopen(file, 'r');
txt = char(fread(fid, Inf, '*uint8')');      % bytes in, bytes out: no re-encoding
fclose(fid);
for ks = 1:size(pairs,1)
    assert(numel(strfind(txt, pairs{ks,1})) == 1, ...
        'one_factor_patch: expected exactly one %s in %s', pairs{ks,1}, rel);
    txt = strrep(txt, pairs{ks,1}, pairs{ks,2});
end
fid = fopen(file, 'w');
fwrite(fid, uint8(txt));
fclose(fid);
end
