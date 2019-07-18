function [L,dL,ffmat] =  StateSpaceModelsofSpikeTrains(ff,yymat,cufx,cuu,sigma2)
[nt,nneur] = size(yymat);
ffmat = reshape(ff,[],nneur);
ff = vec(ffmat);
yy = vec(yymat);
maxff = max(ff);
ff1 = ff-maxff;
log_yy_ff = yy'*ff-sum(exp(ff1))*exp(maxff);
% log_ff = -0.5*trace(ffmat'*pdinv(cufx'*cuuinv*cufx+sigma2*eye(size(cufx,2)))*ffmat);

% Quadratic term
% cuu = pdinv(cuuinv);
invcc = pdinv(cufx*cufx'+sigma2*cuu);
cf = cufx*ffmat;
log_ff = -.5*trace(ffmat'*ffmat)/sigma2+.5*trace(invcc*cf*cf')/sigma2;

L = log_yy_ff+log_ff;
L = -L;

%%
dL11 = yy-exp(ff1)*exp(maxff);
dL2 = -vec(ffmat/sigma2-cufx'*invcc*(cufx*ffmat)/sigma2);
dL11 = reshape(dL11,[],nneur);
dL1 = vec(dL11);
dL = dL1+dL2;
dL = -dL;

