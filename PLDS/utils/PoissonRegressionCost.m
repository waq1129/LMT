function [f,df] =  PoissonRegressionCost(vecD,y,u,lam,over_m,over_v)
%
% [f,df] =  PoissonRegressionCost(vecD,y,u,lam,over_m,over_v)
%
% Poisson regression cost funtion
% 
%

[yDim T] = size(y);
uDim     = size(u,1);

D    = reshape(vecD,yDim,uDim);
Du   = D*u;
nu   = Du;
if ~isempty(over_m); nu = nu + over_m; end
yhat = nu;
if ~isempty(over_m); yhat = yhat + 0.5*over_v;end
yhat = exp(yhat);

f  = sum(vec(-y.*nu+yhat));
df = (yhat-y)*u';

% L2 regularization
f  = f  + 0.5*lam*norm(D,'fro').^2;
df = df + lam*D;

df = vec(df);
