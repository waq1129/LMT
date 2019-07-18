function [L,dL] = logmargli_gplvm_se_block_ref_nogrid(uu,xgrid,invkf,yymat,BBwfun,covfun,nf,BBwTfun,ntr)
uux = reshape(uu,ntr,[],nf);
xxsamp = BBwfun(uux,0);

%%%%%%% cov %%%%%%%%
% [kxx,dcc] = covfun(xgrid,xgrid);
% nsevar = kxx(1,1)*sigma2;
% invkxx = pdinv(kxx+nsevar*eye(size(kxx)));

% poisson
xxsamp_mt = reshape(xxsamp,[],nf);
[ctx,dctx_x] = covfun(xgrid,xxsamp_mt);
ctx = ctx';
ffmat = ctx*invkf;

ff = vec(ffmat);
yy = vec(yymat);
log_yy_ff = yy'*ff-sum(exp(ff));

L = -log_yy_ff+.5*trace(uu'*uu);
% [Qtrm,logdettrm,.5*trace(uu'*uu),L]

%% kux
dcufx = dctx_x;
dL_Kuf = -invkf*yymat'+invkf*exp(ffmat)';
dL_K1 = repmat(dL_Kuf,nf,1);
dKuf1 = reshape(dcufx,[],size(xxsamp_mt,1));
dcf = dL_K1.*dKuf1;
dcf1 = sum(reshape(dcf,size(xgrid,1),[]),1);
dL_c3 = reshape(dcf1,nf,[])';
dL_c3 = reshape(dL_c3,ntr,[],nf);

dL_ux = vec(BBwTfun(dL_c3,0));

dL = dL_ux+vec(uu);

