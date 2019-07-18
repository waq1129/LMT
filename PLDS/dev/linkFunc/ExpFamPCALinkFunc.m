function [C, X, d] = ExpFamPCALinkFunc(Y,xDim,varargin)
%
% [C, X, d] = ExpFamPCALinkFunc(Y,xDim,varargin)
%
% exponential family PCA, currently only implemented for
% exponential-Poisson observation model, i.e. learns C, X and d for model 
% Y ~ Poisson(exp(Cx+d+s));
%
% inputs:
% Y:     matrix of size yDim x T
% s:     additive , observed input, same size as Y or scalar (optional)
% xDim:  scalar, dimensionality of latent space
%
% output: 
% C:      loading matrix, of size yDim x xDim
% X:      recovered latent factors, of size xDim x T
% d:      mean offset
%
%
% (c) Lars Buesing 2014
%
% changes: needs linkFunc and dlinkFunc
%

linkFunc            = @(x) log(1+exp(x));
dlinkFunc           = @(x) 1./(1+exp(-x));

s                   = 0;
lam                 = 1e-1;  % penalizer
CposConstrain       = false; % constrain C to be have pos elements

options.display     = 'none';
options.MaxIter     = 10000;
options.maxFunEvals = 50000;
options.Method      = 'lbfgs';
options.progTol     = 1e-9;
options.optTol      = 1e-5;

Cinit = [];
Xinit = [];
dinit = [];

assignopts(who,varargin);

[yDim T] = size(Y);

%% rough initialization for ExpFamPCA based on SVD
if isempty(Cinit)
  my = mean(Y-s,2);
  [Uy Sy Vy] = svd(bsxfun(@minus,Y-s,my),'econ');
  my = max(my,0.1);

  Cinit = Uy(:,1:xDim);
  if CposConstrain
    Cinit = 0.1*randn(yDim,xDim);
  end
  %disp('ExpFamPCA: empty Cinit, do PCA')
end
if isempty(Xinit); Xinit = 0.01*randn(xDim,T);end;
if isempty(dinit); dinit = log(my); end

CXdinit = [vec([Cinit; Xinit']); dinit];

%run ExpFamCPA  
CXdOpt  = minFunc(@ExpFamPCALinkFuncCost,CXdinit,options,Y,xDim,lam,s,CposConstrain,linkFunc,dlinkFunc); 

% Function returns all parameters lumped together as one vector, so need to disentangle: 
d  = CXdOpt(end-yDim+1:end);
CX = reshape(CXdOpt(1:end-yDim),yDim+T,xDim);
C  = CX(1:yDim,:);
if CposConstrain; C = exp(C); end
X  = CX(yDim+1:end,:)';
Xm = mean(X,2);
X  = bsxfun(@minus,X,Xm);
d  = d+C*Xm;

if ~CposConstrain
  % transform C to have orthonormal columns
  [UC SC VC] = svd(C);
  M = SC(1:xDim,1:xDim)*VC(:,1:xDim)';
  C = C/M;
  X = M*X;

  % transform for X to have orthogonal rows
  [MU MS MV] = svd((X*X')./T);
  M = MU';
  C = C/M;
  X = M*X;
end