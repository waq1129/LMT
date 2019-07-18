function D = PoissonRegression(Y,U,varargin)
%
% D = PoissonRegression(Y,U,varargin)
%
% Y  ~  Poisson(D*U+x)
%
% where x ~ Normal(over_m,V) with over_v = diag(V); 
%
% 
% INPUT:
%
% - data Y
% - observed variates U
% - lam:     penalizer for L2 regularization of D  [optional, default = 0.1]
% - over_m:  mean of x   [optional, default 0]
% - over_v:  diagonal of covariance of x  [optional, default 0]
% - Dinit:   inital value of D  [optinal, default 0]
% - options: minFunc optimzation options [optinal]
%
% L Buesing, 2014


yDim = size(Y,1);
uDim = size(U,1);

lam    = 0.1;
over_m = [];
over_v = [];
Dinit  = zeros(yDim,uDim);

options.Display     = 'iter';
options.Method      = 'lbfgs';
options.MaxIter     = 5000;
options.maxFunEvals = 50000;
options.progTol     = 1e-9;
options.optTol      = 1e-5;

assignopts(who,varargin);


D = minFunc(@PoissonRegressionCost,vec(Dinit),options,Y,U,lam,over_m,over_v);
D = reshape(D,yDim,uDim);


