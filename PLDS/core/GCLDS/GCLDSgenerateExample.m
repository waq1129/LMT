function params = GCLDSgenerateExample(varargin)
%
% trueparams = GCLDSgenerateExample(varargin)
%
% generate a random GCLDS model based on some inputs
%
% Yuanjun Gao, 2015

params   = [];
useR     = false;
uDim     = 0;
xDim     = 10;
yDim     = 100;

Aspec    = 0.99;
Arand    = 0.03;
Q0max    = 0.3;
BernFlag = false;
doff     = -1.9;
statFlag = false;
Bscale   = 1.0;

K = 8;
g = (1:K)*doff; 

assignopts(who,varargin);


%%%%%%%%%  generate parameters %%%%%%%%% 

A  = eye(xDim)+Arand*randn(xDim);
A  = A./max(abs(eig(A)))*Aspec; %ensure a stationary distribution
Q  = diag(rand(xDim,1));
Q0 = dlyap(A,Q); %stationary distribution of the series
M  = diag(1./sqrt(diag(Q0)));
A  = M*A*pinv(M);
Q  = M*Q*M'; Q=(Q+Q)/2; %this makes stationary dist has equal variance

O  = orth(randn(xDim));
Q0 = O*diag(rand(xDim,1)*Q0max)*O'/3; %start with something of low variance
x0 = randn(xDim,1)/3;

C  = randn(yDim,xDim)./sqrt(3*xDim);
%d  = 0.3*randn(yDim,1)+doff;

if size(g, 1) ~= yDim,
    g = reshape(g, 1, []);
    g = repmat(g, yDim, 1);
end

params.model.A    = A;
params.model.Q    = Q;
params.model.Q0   = Q0;
params.model.x0   = x0;
params.model.C    = C;
%params.model.d    = d;
params.model.g    = g;

%if BernFlag
%    params.model.dualHandle = @LogisticBernoulliDualHandle;
%    params.model.likeHandle = @LogisticBernoulliHandle;
%end

if uDim>0
  cQ = max(abs(diag(chol(params.model.Q))));
  params.model.B = cQ*(rand(xDim,uDim)+0.5)/(uDim)*Bscale;
  params.model.notes.useB = true;
end

%if statFlag
%  params.model.x0 = zeros(xDim,1);
%  params.model.Q0 = dlyap(params.model.A,params.model.Q);
%end

params = GCLDSsetDefaultParameters(params,xDim,yDim,K);
