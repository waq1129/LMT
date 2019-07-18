function params = PLDSLinkFuncgenerateExample(varargin)
%
% function params = PLDSLinkFuncgenerateExample(varargin)
%


% !!! should we reuse PLDSgenerateExample here? 

params   = [];
useR     = false;
uDim     = 0;
xDim     = 10;
yDim     = 100;

Aspec    = 0.99;
Arand    = 0.03;
Q0max    = 0.3;
doff     = -1.9;
statFlag = false;
Bscale   = 1.0;

assignopts(who,varargin);


%%%%%%%%%  generate parameters %%%%%%%%% 

A  = eye(xDim)+Arand*randn(xDim);
A  = A./max(abs(eig(A)))*Aspec;
Q  = diag(rand(xDim,1));
Q0 = dlyap(A,Q);
M  = diag(1./sqrt(diag(Q0)));
A  = M*A*pinv(M);
Q  = M*Q*M'; Q=(Q+Q)/2;

O  = orth(randn(xDim));
Q0 = O*diag(rand(xDim,1)*Q0max)*O'/3;
x0 = randn(xDim,1)/3;

C  = randn(yDim,xDim)./sqrt(3*xDim);
d  = 0.3*randn(yDim,1)+doff;

params.model.A    = A;
params.model.Q    = Q;
params.model.Q0   = Q0;
params.model.x0   = x0;
params.model.C    = C;
params.model.d    = d;

if uDim>0
  cQ = max(abs(diag(chol(params.model.Q))));
  params.model.B = cQ*(rand(xDim,uDim)+0.5)/(uDim)*Bscale;
  params.model.notes.useB = true;
end

if statFlag
  params.model.x0 = zeros(xDim,1);
  params.model.Q0 = dlyap(params.model.A,params.model.Q);
end

params = PLDSLinkFuncsetDefaultParameters(params,xDim,yDim);
