function params = LDSgenerateExample(varargin)
%
% params = generateLDS(varargin)
%
% make a random LDS given some parameters

%NOT DOCUMENTED YET%

uDim     = 0;
xDim     = 10;
yDim     = 100;

Arot     = 0.1;
Aspec    = 0.99;
Arand    = 0.03;
Q0max    = 0.3;
Rmin     = 0.1;
Rmax     = 0.1;

assignopts(who,varargin);


%%%%%%%%%  generate parameters %%%%%%%%% 

A  = eye(xDim)+Arand*randn(xDim);
A  = A./max(abs(eig(A)))*Aspec;
MAS = randn(xDim); MAS = (MAS-MAS')/2;A  = expm(Arot.*(MAS))*A;
Q  = diag(rand(xDim,1));
Q0 = dlyap(A,Q);
M  = diag(1./sqrt(diag(Q0)));
A  = M*A*pinv(M);
Q  = M*Q*M'; Q=(Q+Q)/2;

O  = orth(randn(xDim));
Q0 = O*diag(rand(xDim,1)*Q0max)*O'/3;
x0 = randn(xDim,1)/3;

C  = randn(yDim,xDim)./sqrt(3*xDim);
R  = diag(rand(yDim,1)*Rmax+Rmin);
d  = 0.3*randn(yDim,1);

params.model.A    = A;
params.model.Q    = Q;
params.model.Q0   = Q0;
params.model.x0   = x0;
params.model.C    = C;
params.model.d    = d;
params.model.R    = R;
params.model.Pi   = dlyap(params.model.A,params.model.Q);
params.model.notes.useR = true;
params.model.notes.useS = false;

if uDim>0
  cQ = max(abs(diag(chol(params.model.Q))));
  params.model.B    = cQ*(rand(xDim,uDim)+0.5)/(uDim);
  params.model.notes.useB = true;
else 
  params.model.notes.useB = false;
end

params = LDSsetDefaultParameters(params,xDim,yDim);

