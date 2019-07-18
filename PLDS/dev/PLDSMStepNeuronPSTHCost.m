function [f,df] =  PLDSMStepNeuronPSTHCost(W,ydat,mdat,vdat,lam)
%

Trials   = size(ydat,2);
slowFlag = 1;
nu   = bsxfun(@plus,mdat,W);
yhat = exp(nu+0.5*vdat);

f  = sum(vec(-ydat.*nu+yhat))/Trials;
df = sum(-ydat+yhat,2)/Trials;

if ~slowFlag
  f  = f +lam*0.5*norm(W,'fro').^2;
  df = df+lam*W;
else
  dW = W(1:end-1)-W(2:end);
  f = f + 0.5*lam*norm(dW,'fro').^2;
  rdf = zeros(size(W));
  rdf(1:end-1) = rdf(1:end-1) + dW;
  rdf(2:end)   = rdf(2:end) - dW;
  df = df + lam*rdf;
end