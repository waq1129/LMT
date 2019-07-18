clear all
close all

addpath('/nfs/nhome/live/lars/projects/dynamics/NucNormSubspace/ChangePointNucNormMin');

xDim   = 10;
yDim   = 100;

T      = 100; 
Trials = numel(T);

%options.display     = 'none';
options.MaxIter     = 10000;
options.maxFunEvals = 50000;
options.Method      = 'lbfgs';
options.progTol     = 1e-9;
options.optTol      = 1e-5;


[C d S X] = sampleSwitchingSubspaces(xDim,yDim,1,Trials,T,'dhom',true);
Y = [S{:}];
C = C{1};
X = [X{:}];
d = d{1};

% transform true parameters
[UC SC VC] = svd(C);
M = SC(1:xDim,1:xDim)*VC(:,1:xDim)';
C = C/M;
X = M*X; 
T = size(Y,2);

mean(vec(Y))

[Cest, Xest, dest] = ExpFamPCA(Y,xDim,'options',options,'dt',1,'lam',1);
subspace(C,Cest) 

Cest'*Cest
Xest*Xest'./T

[UC SC VC] = svd(Cest);
M = SC(1:xDim,1:xDim)*VC(:,1:xDim)';
Cest = Cest/M;
Xest = M*Xest;
U = C'*Cest;
Cest = Cest/U;
Xest = U*Xest;


figure
plot(vec(X),vec(Xest),'rx');

figure
plot(vec(d),vec(dest),'rx'); 
