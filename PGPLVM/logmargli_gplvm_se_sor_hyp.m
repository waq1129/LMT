function L = logmargli_gplvm_se_sor_hyp(loghypxx,loghypxx0,xxsamp,xgrid,latentTYPE,tgrid,nt,hypid,sigma2,ffmat,ffTYPE,cnse)
loghypxx0(hypid) = loghypxx;
BBwfun_x = prior_kernel(exp(loghypxx0(1)),exp(loghypxx0(2)),nt,latentTYPE,tgrid);
% imagesc(Kprior),colorbar
uu = BBwfun_x(xxsamp,1);

nneur = size(ffmat,2);
covfun = covariance_fun(exp(loghypxx0(3)),exp(loghypxx0(4)),ffTYPE); % get the covariance function
cuu = covfun(xgrid,xgrid)+cnse*eye(size(xgrid,1));

%%%%%%% cov %%%%%%%%
cufx = covfun(xgrid,xxsamp);
invcc = pdinv(cufx*cufx'+sigma2*cuu);

% Log-determinant term
logDetS1 = logdetns(cufx*cufx'+sigma2*cuu)-logdetns(cuu)+log(sigma2)*(length(xxsamp)-size(cufx,1));
logdettrm = .5*nneur*logDetS1;

% Quadratic term
cf = cufx*ffmat;
Qtrm = .5*trace(ffmat'*ffmat)/sigma2-.5*trace(invcc*cf*cf')/sigma2;

L = Qtrm+logdettrm+.5*trace(uu'*uu);