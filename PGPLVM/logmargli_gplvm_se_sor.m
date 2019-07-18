% function [L,dL,m0,S0] = logmargli_gplvm_se_sor(uu,BBwfun,ff,covfun,sigma2,nf,BBwTfun,xgrid,cuuinv,covfun_g)
% [nt,nneur] = size(ff);
% if length(uu)==nt*nf
%     opt_m = 0;
%     uu = [uu; zeros(nneur,1)];
% else
%     opt_m = 1;
% end
% 
% % Compute latent
% mm = uu(end-nneur+1:end);
% uu = reshape(uu(1:end-nneur),[],nf);
% xx = BBwfun(uu,0);
% 
% %%%%%%% cov %%%%%%%%
% % [cc,dcc] = covfun_g(xx,xx);
% [cufx,dcufx] = covfun(xgrid,xx);
% cc = cufx'*cuuinv*cufx;
% dcc = cufx'*cuuinv*dcufx;
% 
% % [cc,dcc] = covfun(xx,xx);
% C11 = cc+sigma2*eye(length(xx));
% 
% m0 = mm';
% S0 = C11;
% 
% [invS0, U] = pdinv(S0);
% logDetS0 = logdet(S0, U);
% 
% % Log-determinant term
% logdettrm = .5*nneur*logDetS0;
% 
% % Quadratic term
% Xctr = bsxfun(@minus,ff,m0);  % centered X
% SX = invS0*Xctr;
% Qtrm = .5*trace(Xctr'*SX);
% 
% L = Qtrm+logdettrm+.5*trace(uu'*uu);
% 
% %%
% if nargout>1
%     dL_K = .5*nneur*invS0-.5*SX*SX';
%     dL_K1 = repmat(dL_K,nf,1);
%     dfc = reshape(dcc,[],nt);
%     dcf = dL_K1.*dfc;
%     dcf = sum(reshape(dcf,nt,[]),1);
%     dL_c = 2*reshape(dcf,nf,[])';
%     
%     dL_u = vec(BBwTfun(dL_c,0))+vec(uu);
%     
%     dL_m = vec(-sum(SX,1));
%     if opt_m
%         dL = [dL_u; dL_m];
%     else
%         dL = dL_u;
%     end
% end
% 
function [L,dL] = logmargli_gplvm_se_sor(uu,BBwfun,ff,covfun,sigma2,nf,BBwTfun,xgrid,cuu)
[nt,nneur] = size(ff);
uu = reshape(uu,[],nf);
xx = BBwfun(uu,0);

%%%%%%% cov %%%%%%%%
[cufx,dcufx] = covfun(xgrid,xx);
% cuu = inv(cuuinv);
invcc = inv(cufx*cufx'+sigma2*cuu);

% Log-determinant term
logDetS1 = -logdetns(invcc)-logdetns(cuu)+log(sigma2)*(length(xx)-size(cufx,1));
logdettrm = .5*nneur*logDetS1;

% Quadratic term
cf = cufx*ff;
Qtrm = .5*trace(ff'*ff)/sigma2-.5*trace(invcc*cf*cf')/sigma2;

L = Qtrm+logdettrm+.5*trace(uu'*uu);

%%
dL_Kuf_logtrm = nneur*invcc*cufx;
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



