function [f, df] = PLDSMStepObservationCostWithR(vecCd,seq,params)
%
% function [f, df] = PLDSMStepObservationCost(vecCd,seq,params)
%
% Mstep for observation parameters C,d for standard PLDS with exp-Poisson observations
%
% Input:
%	- convention Cd = [C d]  and vecCd = vec(Cd)
%
% to do: 
%
%       0) analyze run time
%
%
% (c) L Buesing 2014


Trials  = numel(seq);
yDim    = size(seq(1).y,1);
xDim    = size(params.model.A,1);


CdMat   = reshape(vecCd,yDim,xDim+1);
C       = CdMat(:,1:xDim);
d       = CdMat(:,end);

CC      = zeros(yDim,xDim^2);
for yd=1:yDim
  CC(yd,:) = vec(C(yd,:)'*C(yd,:));
end


f   = 0;				% current value of the cost function = marginal likelihood
df  = zeros(size(C));			% derviative wrt C
dfd = zeros(yDim,1);			% derivative wrt d

for tr=1:Trials
 
  T    = size(seq(tr).y,2);
  y    = seq(tr).y;
  m    = seq(tr).posterior.xsm;
  Vsm  = reshape(seq(tr).posterior.Vsm',xDim.^2,T);
     
  h    = bsxfun(@plus,C*m,d);
  if params.model.notes.useS; h = h+seq(tr).s; end
  rho  = CC*Vsm;

  if params.model.notes.useR
    h    = h+seq(tr).posterior.n_ast;
    R    = repmat(params.model.R,1,T);
    Rlam = R.*reshape(seq(tr).posterior.lamOpt,yDim,T);
    Rbar = R./(Rlam+1);
    rho  = rho./((Rlam+1).^2)+Rbar;
  end

  yhat = exp(h+rho/2);

  f    = f+sum(vec(y.*h-yhat));
  
  TT   = yhat*Vsm';
  TT   = reshape(TT,yDim*xDim,xDim);
  TT   = squeeze(sum(reshape(bsxfun(@times,TT,vec(C)),yDim,xDim,xDim),2));
     
  df   = df  + (y-yhat)*m'-TT;
  dfd  = dfd + sum((y-yhat),2);
  
end

f  = -f;
df = -vec([df dfd]);
