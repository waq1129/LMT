function [L,dL,U] = logmargli_gplvm_se_sor_var(vv_x,U,yy,BBwfun,covfun,sigma2,nf,BBwTfun,xgrid,cuuinv,pp)
[nt,nneur] = size(yy);
vv_x = reshape(vv_x,[],nf);
xx = BBwfun(vv_x,0);

U = reshape(U,[],nneur);

%%%%%%% cov %%%%%%%%
[Kuf,dKuf] = covfun(xgrid,xx);
% cuuinv = pdinv(cuu);
cuuinvhalf = chol(cuuinv);
Kuu_uf = cuuinv*Kuf;
dfu = sum((cuuinvhalf*Kuf).^2,1)';
ukuf = U'*Kuu_uf;
ucuinv = U'*cuuinv;

rhoff = covfun(1,1)+sigma2;
exp_rho = exp(rhoff/2);
one_vec = ones(nneur,1);
exp_dfu = exp(-0.5*dfu);

%%
trm1 = trace(ukuf*yy);

trm2 = -exp_rho*one_vec'*exp(ukuf)*exp_dfu;

trm3 = -0.5*trace(ucuinv*U)-0.5*trace(vv_x'*vv_x);

L = trm1+trm2+trm3;
L = -L;

%%
switch pp
    case 1 % X
        dL_Kuf_trm1 = ucuinv'*yy';
        dL_Kuf_trm2 = -exp_rho*ucuinv'*bsxfun(@times,exp(ukuf),exp_dfu') ...
            +exp_rho*cuuinvhalf'*bsxfun(@times,(exp(ukuf')*one_vec).*exp_dfu,(cuuinvhalf*Kuf)')';
        dL_Kuf = dL_Kuf_trm1+dL_Kuf_trm2;
        
        dL_K1 = repmat(dL_Kuf,nf,1);
        dKuf1 = reshape(dKuf,[],nt);
        
        dcf = dL_K1.*dKuf1;
        dcf1 = sum(reshape(dcf,size(xgrid,1),[]),1);
        dL_c = reshape(dcf1,nf,[])';
        
        dL_x = vec(BBwTfun(dL_c,0))-vec(vv_x);
        
        dL = -dL_x;
    case 2 % U
        dL = Kuu_uf*yy-exp_rho*Kuu_uf*bsxfun(@times,exp(ukuf),exp_dfu')'-ucuinv';
        dL = -vec(dL);
end

