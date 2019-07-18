function L = logmargli_gplvm_se_sor_hyp(loghypxx,loghypxx0,xxsamp,yymat,xgrid,latentTYPE,tgrid,nt,hypid,sigma2,fgrid,ffTYPE,ntr)
loghypxx0(hypid) = loghypxx;

%%
nf = size(xxsamp,3);
xxsamp = reshape(xxsamp,ntr,[],nf);
[Bfun, BTfun, nu] = prior_kernel_sp(exp(loghypxx0(1)),exp(loghypxx0(2)),nt,latentTYPE,tgrid);
Bfun = @(x,f) permute(reshape(Bfun(reshape(permute(x,[2,1,3]),size(x,2),[]),f),[],size(x,1),size(x,3)),[2,1,3]);
uu = vec(Bfun(xxsamp,1));

%%%%%%% cov %%%%%%%%
covfun = covariance_fun(exp(loghypxx0(3)),exp(loghypxx0(4)),ffTYPE); % get the covariance function
[kxx,dcc] = covfun(xgrid,xgrid);
sigma2 = kxx(1,1)*sigma2;
invkxx = pdinv(kxx+sigma2*eye(size(kxx)));

% poisson
xxsamp_mt = reshape(xxsamp,[],nf);
ctx = covfun(xgrid,xxsamp_mt);
ctx = ctx';
invkf = invkxx*fgrid;
ffmat = ctx*invkf;

ff = vec(ffmat);
yy = vec(yymat);
log_yy_ff = yy'*ff-sum(exp(ff));

L = -log_yy_ff+.5*trace(uu'*uu);
