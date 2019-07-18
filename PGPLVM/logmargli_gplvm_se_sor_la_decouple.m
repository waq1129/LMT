function [L,dL] = logmargli_gplvm_se_sor_la_decouple(uu,yy,BBwfun,ff,covfun,sigma2,nf,BBwTfun,xgrid,cuu,cuuinv,cufx_old,invcc_old)
[nt,nneur] = size(ff);
uu = reshape(uu,[],nf);
xx = BBwfun(uu,0);

%%%%%%% cov %%%%%%%%
[cufx,dcufx] = covfun(xgrid,xx);
% cuuinv = pdinv(cuu);
eff = exp(ff).*ff;

% Log-determinant term
logdetB = 0;
ytrm = 0;
etrm = 0;
qtrm = 0;
dL_logdet = 0;
dL_ytrm = 0;
dL_qtrm = 0;
dL_etrm = 0;
for nn=1:nneur
    
    fn = ff(:,nn);
    efn = eff(:,nn);
    yn = yy(:,nn);

    W = exp(vec(fn));
    dd = W*sigma2+1;
    dw = sigma2+1./W;
    invdw = 1./dw;
    invwcf = bsxfun(@times,invdw,cufx');
    cwc = cufx*invwcf+cuu;
    invcwc = pdinv(cwc);
    logdetB = logdetB+logdetns(cwc)-logdetns(cuu)+sum(log(dd));
    
    dL_logdet = dL_logdet+invcwc*invwcf';
    
    %%
    %     ff0 = invdd.*fn/sigma2-invdd.*cufx_old'*invcc_old*cufx_old*fn/sigma2+invdd.*efn...
    %         trm1 = -invdw*cufx'*invcwc*cufx*invdd.*fn/sigma2...
    %         trm2 = +invdw*cufx'*invcwc*cufx*invdd.*cufx_old'*invcc_old*cufx_old/sigma2*fn...
    %         trm3 = -invdw*cufx'*invcwc*cufx*invdd.*efn;
    
    invdd = 1./dd;
    ftrm1 = cinvc(cufx,invcwc,invdw,invdd.*fn/sigma2);
    tmp = invdd.*(cufx_old'*(invcc_old*(cufx_old*fn))/sigma2);
    ftrm2 = cinvc(cufx,invcwc,invdw,tmp);
    ftrm3 = cinvc(cufx,invcwc,invdw,invdd.*efn);
    
    ftrm0 = invdd.*fn/sigma2-tmp+invdd.*efn;
    ff0 = ftrm0-ftrm1+ftrm2-ftrm3;
    
    %%
    cf0 = cufx*ff0;
    cc = cuuinv*cufx;
    ef0 = exp(cc'*cf0+sigma2*ff0);
    cfyn = cufx*yn;
    cfef0 = cufx*ef0;
    ccf0 = cuuinv*cf0;
    
    cy = cc'*cfyn+sigma2*yn;
    ce = cc'*cfef0+sigma2*ef0;
    cff = cc'*cf0+sigma2*ff0;
    [~,yftrm1,dftrm1,deftrm1,dqftrm1] = cinvc(cufx,invcwc,invdw,invdd.*fn/sigma2,cy',ce',cff');
    [~,yftrm2,dftrm2,deftrm2,dqftrm2] = cinvc(cufx,invcwc,invdw,tmp,cy',ce',cff');
    [~,yftrm3,dftrm3,deftrm3,dqftrm3] = cinvc(cufx,invcwc,invdw,invdd.*efn,cy',ce',cff');
    
    % y*f
    ytrm = ytrm+cy'*ftrm0-yftrm1+yftrm2-yftrm3;
    dL_ytrm = dL_ytrm+ccf0*yn'+(cc*yn)*ff0'-dftrm1+dftrm2-dftrm3;
    
    % exp(f)
    etrm = etrm + sum(exp(cufx'*(cc*ff0)+sigma2*ff0));
    dL_etrm = dL_etrm+ccf0*ef0'+(cc*ef0)*ff0'-deftrm1+deftrm2-deftrm3;
    
    % qtrm
    qtrm = qtrm + cff'*ff0;
    dL_qtrm = dL_qtrm+.5*ccf0*ff0'+.5*(cc*ff0)*ff0'-dqftrm1+dqftrm2-dqftrm3;
    
end

L = 0.5*logdetB-ytrm+etrm+.5*qtrm+.5*trace(uu'*uu);

%%
dL_Kuf = dL_logdet-dL_ytrm+dL_etrm+dL_qtrm;

dL_K1 = repmat(dL_Kuf,nf,1);
dKuf1 = reshape(dcufx,[],nt);

dcf = dL_K1.*dKuf1;
dcf1 = sum(reshape(dcf,size(xgrid,1),[]),1);
dL_c = reshape(dcf1,nf,[])';

dL_u = vec(BBwTfun(dL_c,0))+vec(uu);

dL = dL_u;
