function [L,dL,ffmat,log_yy_ff] =  StateSpaceModelsofSpikeTrains_ref(ff,yymat,invcxx,kk)
[nt,nneur] = size(yymat);
ffmat0 = reshape(ff,[],nneur);
ffmat = kk*ffmat0;

ff = vec(ffmat);
yy = vec(yymat);
log_yy_ff = yy'*ff-sum(exp(ff));

% Quadratic term
log_ff = -.5*trace(invcxx*ffmat0*ffmat0');

L = log_yy_ff+log_ff;
L = -L;

% [log_yy_ff,log_ff]
%%
dL11 = yy-exp(ff);
dL11 = reshape(dL11,[],nneur);
dL1 = vec(kk'*dL11);
dL2 = -vec(invcxx*ffmat0);
dL = dL1+dL2;
dL = -dL;