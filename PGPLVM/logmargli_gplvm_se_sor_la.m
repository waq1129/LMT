function [L,dL] = logmargli_gplvm_se_sor_la(uu,BBwfun,ff,covfun,sigma2,nf,BBwTfun,xgrid,cuu)
[nt,nneur] = size(ff);
uu = reshape(uu,[],nf);
xx = BBwfun(uu,0);

%%%%%%% cov %%%%%%%%
[cufx,dcufx] = covfun(xgrid,xx);
invcc = pdinv(cufx*cufx'+sigma2*cuu);

% Log-determinant term
% logdettrm = 0.5*logdetns(B);
logdetB = 0;
dL_logdet = 0;
for nn=1:nneur
    Whalf = exp(vec(ff(:,nn)/2));
    W = Whalf.^2;
    dd = 1/sigma2+W;
    wdw = W./dd;
    invcuu_wd = pdinv(cuu*sigma2+cufx*bsxfun(@times,wdw,cufx'));
    logdetB = logdetB-logdetns(invcuu_wd)-logdetns(cuu)+sum(log(dd))+log(sigma2)*(nt-size(cufx,1));
    dL_logdet = dL_logdet+invcuu_wd*bsxfun(@times,wdw',cufx);
end
logdettrm = 0.5*logdetB;

% Quadratic term
cf = cufx*ff;
Qtrm = .5*trace(ff'*ff)/sigma2-.5*trace(invcc*cf*cf')/sigma2;

L = Qtrm+logdettrm+.5*trace(uu'*uu);

%%
dL_Kuf_logtrm = dL_logdet;
dL_Kuf_qtrm1 = invcc*(cf*cf')*invcc*cufx/sigma2;
dL_Kuf_qtrm2 = -invcc*cf*ff'/sigma2;
dL_Kuf = dL_Kuf_qtrm1+dL_Kuf_qtrm2+dL_Kuf_logtrm;
dL_K1 = repmat(dL_Kuf,nf,1);
dKuf1 = reshape(dcufx,[],nt);

dcf = dL_K1.*dKuf1;
dcf1 = sum(reshape(dcf,size(xgrid,1),[]),1);
dL_c = reshape(dcf1,nf,[])';

dL_u = vec(BBwTfun(dL_c,0))+vec(uu);

dL = dL_u;
