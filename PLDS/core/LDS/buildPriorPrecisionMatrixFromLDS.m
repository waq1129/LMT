function Lambda = buildPriorPrecisionMatrixFromLDS(params,T)
%
% Lambda = buildPrecisionMatrixFromLDS(params,T)
%
% construct the precision matrix of the prior across all time points and
% dimensions, as described in Paninski et al, A new look at state-space
% models for neural data, 2009
%
% c/o L Buesing and J Macke, 01/2014


xDim   = size(params.model.A,1);
invQ   = pinv(params.model.Q);
invQ0  = pinv(params.model.Q0);
AinvQ  = params.model.A'*invQ;
AinvQA = AinvQ*params.model.A;


Lambda = sparse(T*xDim,T*xDim);
Lambda(1:xDim,1:xDim) = invQ0;

for t=1:T-1
  xidx = ((t-1)*xDim+1):(t*xDim);
  Lambda(xidx,xidx) = Lambda(xidx,xidx)+AinvQA;
  Lambda(xidx,xidx+xDim) = -AinvQ;
  Lambda(xidx+xDim,xidx) = -AinvQ';
  Lambda(xidx+xDim,xidx+xDim) = Lambda(xidx+xDim,xidx+xDim)+invQ;
end
Lambda = sparse((Lambda+Lambda')/2);
