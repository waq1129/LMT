function L = logmargli_gplvm_se_sor_var_hyp(loghypxx,loghypxx0,xxsamp,U,yy,xgrid,latentTYPE,tgrid,hypid,ffTYPE)
[nt,nneur] = size(yy);

loghypxx0(hypid) = loghypxx;
BBwfun_x = prior_kernel(exp(loghypxx0(1)),exp(loghypxx0(2)),nt,latentTYPE,tgrid);
vv_x = BBwfun_x(xxsamp,1);
sigma2 = exp(loghypxx0(5));

%%%%%%% cov %%%%%%%%
covfun = covariance_fun(exp(loghypxx0(3)),exp(loghypxx0(4)),ffTYPE); % get the covariance function
cuu = covfun(xgrid,xgrid)+sigma2*eye(size(xgrid,1));

Kuf = covfun(xgrid,xxsamp);
cuuinv = pdinv(cuu);
cuuinvhalf = chol(cuuinv);
Kuu_uf = cuuinv*Kuf;
dfu = sum((cuuinvhalf*Kuf).^2,1)';
ukuf = U'*Kuu_uf;
ucuinv = U'*cuuinv;

rhoff = covfun(1,1)+sigma2;
exp_rho = exp(rhoff/2);
one_vec = ones(nneur,1);
exp_dfu = exp(-0.5*dfu);

trm1 = trace(ukuf*yy);

trm2 = -exp_rho*one_vec'*exp(ukuf)*exp_dfu;

trm3 = -0.5*trace(ucuinv*U)-0.5*trace(vv_x'*vv_x);

L = trm1+trm2+trm3;
L = -L;
