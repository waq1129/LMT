function [f df ddf] = PLDSLaplaceCost(x,y,Lambda,mu,W,d)
%
%
%
over_m = W*x+d;
yhat   = exp(over_m);
xmu    = x-mu;
Lamxmu = Lambda*mu;
yhaty  = yhat-y;

f   = 0.5*xmu'*Lamxmu + y'*over_m-sum(yhat);

df  = Lamxmu + W'*(yhat-y);

ddf = Lambda + W'*diag(yhat-y)*W;