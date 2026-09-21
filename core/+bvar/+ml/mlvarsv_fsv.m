% bvar.ml.mlvarsv_fsv - log marginal likelihood of the VAR-FSV model (factor SV)
% by adaptive importance sampling, with the VAR coefficients and the r latent
% factors integrated out analytically and the remaining parameters drawn from
% importance densities fitted to the posterior draws.
%
%   [lml,lmlstd,out] = bvar.ml.mlvarsv_fsv(X,Y,Y0,M,Hyper,flag_marg,store_h,...
%       store_hpara,store_l,store_kappa,is_kappafixed,is_kappasym,'gram','full')
%
%   M           - number of importance draws, rounded up here to a multiple of
%                 50; lml is the mean of the 50 batch log marginal likelihoods
%                 and lmlstd their standard error
%   flag_marg   - 1 or 2; any other value raises an error. 2 also integrates out
%                 the log-volatility variances and is what run_ml passes by
%                 default; VAR-FSV is the only model in this family that
%                 implements 1
%   Hyper: alp0, Valp, c0, nuh, Sh, mu0, Vmu, phi0, Vphi, l0, Vl. Valp is
%          recomputed inside from each kappa draw before any read, so whichever
%          version the caller passes is irrelevant.
%   store_h     - nsim x T x (n+r); r is inferred as size(store_h,3)-n
%   store_hpara - nsim x 3(n+r), columns [mu' phi' sig2']
%   store_l     - nsim x kl free loadings
%   store_kappa - nsim x 2
%   'gram'      - how the weighted Gram matrix in the precision of the VAR
%                 coefficients is formed: 'full' (default, as the published
%                 code) or 'blocks'. The two agree to rounding and consume the
%                 same random draws; 'blocks' is faster when n is large
%   out: store_w, bigml (the 50 batch values), and the fitted IS parameters
%
% See:
% Chan, J.C.C. (2023). Comparing stochastic volatility specifications for large
% Bayesian VARs, Journal of Econometrics, 235(2), 1419-1446.

function [lml,lmlstd,out] = mlvarsv_fsv(X,Y,Y0,M,Hyper,flag_marg,store_h,...
    store_hpara,store_l,store_kappa,is_kappafixed,is_kappasym,varargin)
gram = 'full';
for iv = 1:2:numel(varargin)
    switch lower(varargin{iv})
        case 'gram', gram = varargin{iv+1};
        otherwise, error('bvar:ml:mlvarsv_fsv:badOption', ...
                'unknown option ''%s''', varargin{iv});
    end
end
assert(any(strcmp(gram, {'full','blocks'})), 'bvar:ml:mlvarsv_fsv:badGram', ...
    'gram must be ''full'' or ''blocks''');
blocks = strcmp(gram, 'blocks');
assert(isequal(flag_marg,1) || isequal(flag_marg,2), 'bvar:ml:mlvarsv_fsv:flagMarg', ...
    'flag_marg must be 1 or 2 (as in the legacy switch)');

M = 50*ceil(M/50);
[T,n] = size(Y);
npr = size(store_h,3);
r = npr - n;
k = size(X,2);
p = (k-1)/n;
kl = size(store_l,2);
kappa3 = 100;

   % obtain parameters for importance sampling densities
if is_kappafixed
    % do nothing
elseif is_kappasym
    tmp_hat = gamfit(store_kappa(:,1));
    big_kappa = gamrnd(tmp_hat(1),tmp_hat(2),M,1);
    ckappa_hat = [tmp_hat(1); 1/tmp_hat(2)];
else
    ckappa_hat = zeros(2,2);
    big_kappa = zeros(M,2);
    for i=1:2
        tmp_hat = gamfit(store_kappa(:,i));
        big_kappa(:,i) = gamrnd(tmp_hat(1),tmp_hat(2),M,1);
        ckappa_hat(i,:) = [tmp_hat(1); 1/tmp_hat(2)];
    end
end
lhat = mean(store_l)';
lstd = chol(cov(store_l),'lower');
lpre = full(lstd'\(lstd\speye(kl)));
muhat = mean(store_hpara(:,1:n+r))';
muvar = var(store_hpara(:,1:n+r))';
phihat = mean(store_hpara(:,n+r+1:2*(n+r)))';
phivar = var(store_hpara(:,n+r+1:2*(n+r)))';
tmp = zeros(n+r,2);
for i=1:n+r
    tmp(i,:) = gamfit(1./store_hpara(:,2*(n+r)+i));
end
nusig2hat = tmp(:,1); Ssig2hat = 1./tmp(:,2);
h_hat = zeros(T*(n+r),1);
for ii=1:n+r
    [hi_hat,Khi_hat] = bvar.ml.isden_arss(store_h(:,:,ii));
    h_hat((ii-1)*T+1:ii*T) = hi_hat;
    tmp0 = sparse((ii-1)*T,T);
    if ii == 1
        Kh_hat = Khi_hat;
    else
        Kh_hat = [[Kh_hat tmp0]; [tmp0' Khi_hat]];   % block-diagonal, one series at a time
    end
end
CKh_hat = chol(Kh_hat,'lower');

    % obtain importance sampling draws
big_l = repmat(lhat',M,1) + (lstd*randn(kl,M))';
big_mu = repmat(muhat',M,1) + repmat(sqrt(muvar)',M,1).*randn(M,n+r);
big_phi = zeros(M,n+r);
for ii=1:n+r
   big_phi(:,ii) = bvar.util.tnormrnd(phihat(ii),phivar(ii),-1,1,M);
end
big_sig2 = zeros(M,n+r);          % drawn even under flag_marg = 2, where nothing reads it
for i=1:n+r
    big_sig2(:,i) = 1./gamrnd(nusig2hat(i),1./Ssig2hat(i),M,1);
end

switch flag_marg
    case 1
        cprior = -.5*(kl+2*(n+r))*log(2*pi) -sum(log(normcdf((1-Hyper.phi0)./sqrt(Hyper.Vphi)) -normcdf((-1-Hyper.phi0)./sqrt(Hyper.Vphi)))) ...
            -kl/2*log(Hyper.Vl) -.5*sum(log(Hyper.Vmu)) -.5*sum(log(Hyper.Vphi))...
            +Hyper.nuh'*log(Hyper.Sh) -sum(gammaln(Hyper.nuh));
        prior = @(l,m,ph,s) cprior -.5*(l-Hyper.l0)'*(l-Hyper.l0)/Hyper.Vl ...
            -.5*sum((m-Hyper.mu0).^2./Hyper.Vmu) -.5*sum((ph-Hyper.phi0).^2./Hyper.Vphi) ...
            -(Hyper.nuh+1)'*log(s) -sum(Hyper.Sh./s);
        cIS = -.5*(kl+2*(n+r))*log(2*pi) -sum(log(normcdf((1-phihat)./sqrt(phivar)) -normcdf((-1-phihat)./sqrt(phivar)))) ...
            -sum(log(diag(lstd))) -.5*sum(log(muvar))...
            -.5*sum(log(phivar)) +nusig2hat'*log(Ssig2hat) -sum(gammaln(nusig2hat));
        gIS = @(l,m,ph,s) cIS -.5*(l-lhat)'*lpre*(l-lhat) -.5*sum((m-muhat).^2./muvar) ...
            -.5*sum((ph-phihat).^2./phivar) -(nusig2hat+1)'*log(s) -sum(Ssig2hat./s);
    case 2
        if is_kappafixed
            cprior = -.5*(kl+2*(n+r))*log(2*pi) -sum(log(normcdf((1-Hyper.phi0)./sqrt(Hyper.Vphi)) -normcdf((-1-Hyper.phi0)./sqrt(Hyper.Vphi)))) ...
                -kl/2*log(Hyper.Vl) -.5*sum(log(Hyper.Vmu)) -.5*sum(log(Hyper.Vphi));
            prior = @(l,m,ph) cprior -.5*(l-Hyper.l0)'*(l-Hyper.l0)/Hyper.Vl ...
                -.5*sum((m-Hyper.mu0).^2./Hyper.Vmu) -.5*sum((ph-Hyper.phi0).^2./Hyper.Vphi);
            cIS = -.5*(kl+2*(n+r))*log(2*pi) -sum(log(normcdf((1-phihat)./sqrt(phivar)) -normcdf((-1-phihat)./sqrt(phivar)))) ...
                -sum(log(diag(lstd))) -.5*sum(log(muvar)) -.5*sum(log(phivar));
            gIS = @(l,m,ph) cIS -.5*(l-lhat)'*lpre*(l-lhat) -.5*sum((m-muhat).^2./muvar) ...
                -.5*sum((ph-phihat).^2./phivar);
        elseif is_kappasym
            cprior = -.5*(kl+2*(n+r))*log(2*pi) -sum(log(normcdf((1-Hyper.phi0)./sqrt(Hyper.Vphi)) -normcdf((-1-Hyper.phi0)./sqrt(Hyper.Vphi)))) ...
                -kl/2*log(Hyper.Vl) -.5*sum(log(Hyper.Vmu)) -.5*sum(log(Hyper.Vphi))...
                +Hyper.c0(1,1)*log(Hyper.c0(1,2)) -gammaln(Hyper.c0(1,1));
            prior = @(l,m,ph,xk) cprior -.5*(l-Hyper.l0)'*(l-Hyper.l0)/Hyper.Vl ...
                -.5*sum((m-Hyper.mu0).^2./Hyper.Vmu) -.5*sum((ph-Hyper.phi0).^2./Hyper.Vphi) ...
                +(Hyper.c0(1,1)-1)*log(xk) -Hyper.c0(1,2)*xk;
            cIS = -.5*(kl+2*(n+r))*log(2*pi) -sum(log(normcdf((1-phihat)./sqrt(phivar)) -normcdf((-1-phihat)./sqrt(phivar)))) ...
                -sum(log(diag(lstd))) -.5*sum(log(muvar)) -.5*sum(log(phivar)) ...
                +ckappa_hat(1)*log(ckappa_hat(2)) -gammaln(ckappa_hat(1));
            gIS = @(l,m,ph,xk) cIS -.5*(l-lhat)'*lpre*(l-lhat) -.5*sum((m-muhat).^2./muvar) ...
                -.5*sum((ph-phihat).^2./phivar) + (ckappa_hat(1)-1)'*log(xk) -ckappa_hat(2)*xk;
        else
            cprior = -.5*(kl+2*(n+r))*log(2*pi) -sum(log(normcdf((1-Hyper.phi0)./sqrt(Hyper.Vphi)) -normcdf((-1-Hyper.phi0)./sqrt(Hyper.Vphi)))) ...
                -kl/2*log(Hyper.Vl) -.5*sum(log(Hyper.Vmu)) -.5*sum(log(Hyper.Vphi))...
                +Hyper.c0(:,1)'*log(Hyper.c0(:,2)) -sum(gammaln(Hyper.c0(:,1)));
            prior = @(l,m,ph,xk) cprior -.5*(l-Hyper.l0)'*(l-Hyper.l0)/Hyper.Vl ...
                -.5*sum((m-Hyper.mu0).^2./Hyper.Vmu) -.5*sum((ph-Hyper.phi0).^2./Hyper.Vphi) ...
                +(Hyper.c0(:,1)-1)'*log(xk) -Hyper.c0(:,2)'*xk;
            cIS = -.5*(kl+2*(n+r))*log(2*pi) -sum(log(normcdf((1-phihat)./sqrt(phivar)) -normcdf((-1-phihat)./sqrt(phivar)))) ...
                -sum(log(diag(lstd))) -.5*sum(log(muvar)) -.5*sum(log(phivar)) ...
                +ckappa_hat(:,1)'*log(ckappa_hat(:,2)) -sum(gammaln(ckappa_hat(:,1)));
            gIS = @(l,m,ph,xk) cIS -.5*(l-lhat)'*lpre*(l-lhat) -.5*sum((m-muhat).^2./muvar) ...
                -.5*sum((ph-phihat).^2./phivar) + (ckappa_hat(:,1)-1)'*log(xk) -ckappa_hat(:,2)'*xk;
        end
end

store_w = zeros(M,1);
L = [eye(r); ones(n-r,r)];
L_idx = find(tril(ones(n,r),-1)~=0); % index of free elements of L
c_hi = sum(log(diag(CKh_hat)));
for isim = 1:M
    longh = h_hat + CKh_hat'\randn(T*(n+r),1);
    h = reshape(longh,T,n+r);
    l = big_l(isim,:)';
    L(L_idx) = l;
    mu = big_mu(isim,:)';
    phi = big_phi(isim,:)';
    sig2 = big_sig2(isim,:)';
    if is_kappafixed
        kappa = store_kappa(1,:)';
    elseif is_kappasym
        kappa = [big_kappa(isim,1),big_kappa(isim,1)]';
    else
        kappa = big_kappa(isim,:)';
    end
    kappa1 = kappa(1);
    kappa2 = kappa(2);
    [Hyper.alp0,Hyper.Valp] = bvar.priors.minn(p,kappa1,kappa2,kappa3,Y0,Y,4);

    lh_prior = 0;
    for ii=1:n+r
        hi = h(:,ii);
        mui = mu(ii);
        phii = phi(ii);

        Hphii = speye(T)-phii*sparse(2:T,1:(T-1),ones(1,T-1),T,T);
        HiSH = Hphii'*sparse(1:T,1:T,[1-phii^2, ones(1,T-1)])*Hphii;
        switch flag_marg
            case 1
                sigi2 = sig2(ii);
                lh_prior = lh_prior -T/2*log(sigi2) +.5*log(1-phii^2) ...
                    -.5/sigi2*(hi-mui)'*HiSH*(hi-mui);
            case 2
                lh_prior = lh_prior +.5*log(1-phii^2) ...
                    +Hyper.nuh(ii)*log(Hyper.Sh(ii)) -gammaln(Hyper.nuh(ii)) +gammaln(Hyper.nuh(ii)+T/2)...
                    -(Hyper.nuh(ii)+T/2)*log(Hyper.Sh(ii) +.5*(hi-mui)'*HiSH*(hi-mui));
        end
    end
    lh_g = c_hi -.5*(longh-h_hat)'*Kh_hat*(longh-h_hat);
    if blocks
        llike = deny_fsv_blocks(X,Y,L,h,Hyper);
    else
        llike = deny_fsv(X,Y,L,h,Hyper);
    end
    switch flag_marg
        case 1
            store_w(isim) = llike + lh_prior - lh_g ...
                + prior(l,mu,phi,sig2) - gIS(l,mu,phi,sig2);
        case 2
            if is_kappafixed
                store_w(isim) = llike + lh_prior - lh_g ...
                    + prior(l,mu,phi) - gIS(l,mu,phi);
            elseif is_kappasym
                store_w(isim) = llike + lh_prior - lh_g ...
                    + prior(l,mu,phi,kappa1) - gIS(l,mu,phi,kappa1);
            else
                store_w(isim) = llike + lh_prior - lh_g ...
                    + prior(l,mu,phi,kappa) - gIS(l,mu,phi,kappa);
            end
    end
end
shortw = reshape(store_w,M/50,50);
maxw = max(shortw);

bigml = log(mean(exp(shortw-repmat(maxw,M/50,1)),1)) + maxw;
lml = mean(bigml);
lmlstd = std(bigml)/sqrt(50);

out = struct('store_w',store_w, 'bigml',bigml, 'M',M, 'r',r, 'kl',kl, ...
    'h_hat',h_hat, 'Kh_hat',Kh_hat, 'lhat',lhat, 'lstd',lstd, ...
    'muhat',muhat, 'muvar',muvar, 'phihat',phihat, 'phivar',phivar, ...
    'nusig2hat',nusig2hat, 'Ssig2hat',Ssig2hat);
if is_kappafixed
    out.ckappa_hat = [];
else
    out.ckappa_hat = ckappa_hat;
end
end

% -------------------------------------------------------------------------
function lden = deny_fsv(X,Y,L,h,Hyper)
% log p(Y | L, h, kappa) with the VAR coefficients and the latent factors
% integrated out.
[T,n] = size(Y);
npr = size(h,2);
r = npr-n;
bigX = bvar.util.surform2(X,n);
y = reshape(Y',T*n,1);
k_alp = size(bigX,2);
Omega = sparse(1:T*r,1:T*r,reshape(exp(h(:,n+1:n+r))',T*r,1));
Sig = sparse(1:T*n,1:T*n,reshape(exp(h(:,1:n))',T*n,1));

Sy = kron(speye(T),L)*Omega*kron(speye(T),L') + Sig;
CSy = chol(Sy,'lower');
XiSy = ((CSy')\(CSy\bigX))';
Kalp = sparse(1:k_alp,1:k_alp,1./Hyper.Valp) + XiSy*bigX;
CKalp = chol(Kalp,'lower');
tmpc = CKalp\(Hyper.alp0./Hyper.Valp + XiSy*y);
lden = -T*n/2*log(2*pi) -sum(log(diag(CSy))) -.5*sum(log(Hyper.Valp)) -sum(log(diag(CKalp)))...
    -.5*(y'*((CSy')\(CSy\y)) +sum(Hyper.alp0.^2./Hyper.Valp) -sum(tmpc.^2));
end

% -------------------------------------------------------------------------
function lden = deny_fsv_blocks(X,Y,L,h,Hyper)
% deny_fsv with bigX'*inv(Sy)*bigX = sum over t of kron(P_t, x_t'*x_t), where P_t
% is the inverse of the tth n x n block of Sy; Sy and bigX are not formed.
[T,n] = size(Y);
k = size(X,2);
r = size(h,2)-n;
P = zeros(T,n*n);
PY = zeros(T,n);
yPy = 0;
ldSy = 0;
for t=1:T
    CSt = chol(L*diag(exp(h(t,n+1:n+r)))*L' + diag(exp(h(t,1:n))),'lower');
    Pt = (CSt')\(CSt\eye(n));
    P(t,:) = Pt(:)';
    PY(t,:) = Y(t,:)*Pt;
    yPy = yPy + PY(t,:)*Y(t,:)';
    ldSy = ldSy + sum(log(diag(CSt)));
end
XX = reshape(X.*permute(X,[1 3 2]),T,k*k);              % row t is vec(x_t'*x_t)
XPX = reshape(permute(reshape(XX'*P,k,k,n,n),[1 3 2 4]),n*k,n*k);
Kalp = sparse(1:n*k,1:n*k,1./Hyper.Valp) + XPX;
CKalp = chol(Kalp,'lower');
tmpc = CKalp\(Hyper.alp0./Hyper.Valp + reshape(X'*PY,n*k,1));
lden = -T*n/2*log(2*pi) -ldSy -.5*sum(log(Hyper.Valp)) -sum(log(diag(CKalp)))...
    -.5*(yPy +sum(Hyper.alp0.^2./Hyper.Valp) -sum(tmpc.^2));
end
