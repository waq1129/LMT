function Lambda = PLDSLinkFuncbuildPriorPrecisionMatrixFromLDS(params,T,varargin)
%
% Lambda = buildPrecisionMatrixFromLDS(params,T)
%
% construct the precision matrix of the prior across all time points and
% dimensions, as described in Paninski et al, A new look at state-space
% models for neural data, 2009
%
% c/o L Buesing and J Macke, 01/2014

mInd = ones(1,T-1);
assignopts(who,varargin);

xDim   = size(params.model.A,1);
if ~iscell(params.model.Q)
  invQ{1} = pinv(params.model.Q);
else
  for mm=1:numel(params.model.Q)
    invQ{mm} =  pinv(params.model.Q{mm});
  end
end
invQ0  = pinv(params.model.Q0);
for mm=1:numel(invQ)
  AinvQ{mm}  = params.model.A'*invQ{mm};
  AinvQA{mm} = AinvQ{mm}*params.model.A;
end

Lambda = sparse(T*xDim,T*xDim);
Lambda(1:xDim,1:xDim) = invQ0;

for t=1:T-1
  xidx = ((t-1)*xDim+1):(t*xDim);
  Lambda(xidx,xidx) = Lambda(xidx,xidx)+AinvQA{mInd(t)};
  Lambda(xidx,xidx+xDim) = -AinvQ{mInd(t)};
  Lambda(xidx+xDim,xidx) = -AinvQ{mInd(t)}';
  Lambda(xidx+xDim,xidx+xDim) = Lambda(xidx+xDim,xidx+xDim)+invQ{mInd(t)};
end
Lambda = sparse((Lambda+Lambda')/2);
