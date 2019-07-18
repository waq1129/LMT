function [f, df, varBound, m_ast, invV_ast, Vsm, VVsm, over_m, over_v] = VariationalInferenceDualCostGCLDS(lam,VarInfparams)
% 
% [f, df, varBound, m_ast, invV_ast, Vsm, VVsm, over_m, over_v] = VariationalInferenceDualCost(lam,VarInfparams)
%
% Cost function for variational inference via dual optimization for
% Gaussian LDS with exponential family observations
%
% see [M. E. Khan, A. Aravkin, M. Friedlander, and M. Seeger. Fast Dual Variational Inference for Non-Conjugate Latent Gaussian Models. In JMLR W&CP, volume 28, pages 951-959, 2013]
%
% VarInfparams.Lambda
% VarInfparams.y
% VarInfparams.mu
% VarInfparams.W
% VarInfparams.A
% VarInfparams.WlamW
% VarInfparams.g
% VarInfparams.CC
%
% OUTPUT:
% f        = dual cost
% df       = gradient of dual cost wrt lam
% varBound = variational lower bound to marignal log-likelihood log p(y)
% m_ast    = variational posterior mean xDim x T
% invV_ast = variational posterior precision
% Vsm	   = var smoothed cov Cov[x(t),x(t)|y_{1:T}]
% VVsm	   = var smoothed cov Cov[x(t+1),x(t)|y_{1:T}]
% over_m   = W*m_ast+d
% over_v   = diag(W/invV_ast*W)
%
% (c) Lars Buesing, Yuanjun Gao, 2015 
%

xDim     = size(VarInfparams.A,1);
y        = VarInfparams.y; % observed data
[yDim, T] = size(y);
K = size(VarInfparams.g, 2);
yBinary = vec(bsxfun(@eq, vec(VarInfparams.y)', (1:K)'));
g = VarInfparams.g;

% check domain of lam
if ~feval(VarInfparams.domainHandle,lam,K); 
    f = inf; df = nan(size(lam)); 
    return; 
end

W  = VarInfparams.W;          		     	     % loading matrix, sparse, only blk-diag
mu = VarInfparams.mu;         			     % prior mean
%if isfield(VarInfparams,'d')
%  d = VarInfparams.d;		                     % bias for likelihood
%else
  d = repmat(vec(g'), T, 1) - repmat(cumsum(log(1:K)'), T*yDim, 1);
%end
Lambda = VarInfparams.Lambda;			     % precision matrix of LDS prior, assumed to be tri-diagonal

% compute quadratric term of cost function  (lam-y)'*W*Sig*W'*(lam-y) 

lamY          = lam - yBinary;
WlamY         = W'*lamY;
SigWlamY      = Lambda\WlamY;
WlamYSigWlamY = WlamY'*SigWlamY;


%VarInfparams.WlamW  = sparse(zeros(xDim*T)); %debug-line, do note use, already pre-allocated
for t=1:T
  xidx = ((t-1)*xDim+1):(t*xDim);
  yidx = ((t-1)*(yDim*K)+1):(t*(yDim*K));
  %VarInfparams.WlamW(xidx,xidx) = VarInfparams.C'*diag(lam(yidx))*VarInfparams.C; %debug-line, use below
  VarInfparams.WlamW(xidx,xidx) = reshape(VarInfparams.CC*lam(yidx),xDim,xDim);
end
Alam       = Lambda+VarInfparams.WlamW;  % precision matrix of current variational approximation
logdetAlam = logdet_plds(Alam,'chol');

% function value
lam_mat = reshape(lam, K, [])';
[like_f, like_df] = feval(VarInfparams.dualHandle,lam_mat,VarInfparams.dualParams);
like_df = vec(like_df');

dlamY = d .* lamY; %avoid negative infinite d
%dlamY(lamY == 0) = 0;
dlamY(isinf(d)) = 0;
dlamY = sum(dlamY);
f = 0.5*WlamYSigWlamY-mu'*WlamY-dlamY-0.5*logdetAlam+like_f;
  
% catch infeasible lambdas
if isinf(f)
  f = inf; df = nan(size(lam));
  varBound = -inf;
  m_ast = nan(xDim*T,1); invV_ast = nan;
  Vsm = nan(xDim*T,xDim); VVsm = nan(xDim*T,xDim);
  over_m = nan(yDim*T,1); over_v = nan(yDim*T,1);
  return
end

% gradient

CRinvC = zeros(xDim*T,xDim);
for t=1:T
  xidx = ((t-1)*xDim+1):(t*xDim);
  CRinvC(xidx,:) = VarInfparams.WlamW(xidx,xidx);
end

[Vsm, VVsm] = smoothedKalmanMatrices(VarInfparams,CRinvC);
lam_con = zeros(K*yDim*T,1); % equals diag(W/Alam*W')
for t=1:T
  xidx = ((t-1)*xDim+1):(t*xDim);
  yidx = ((t-1)*yDim*K+1):(t*yDim*K);
  %lam_con(yidx) = diag(VarInfparams.C*P(xidx,:)*VarInfparams.C');  %debug-line, use below
  lam_con(yidx) = VarInfparams.CC'*vec(Vsm(xidx,:));
end

%df = W*SigWlamY-W*mu+loglam-0.5*diag(W/Alam*W'); %debug-line, use below
df  = W*SigWlamY-W*mu-d-0.5*lam_con+like_df;
df(isinf(d)) = 0;

%%%%% compute variational lower bound
%if nargout > 2,

m_ast    = mu-SigWlamY;  % optimal mean
invV_ast = Alam;         % optimal inverse covariance

over_m 	 = W*m_ast+d; % T*yDim*K
over_v	 = lam_con;
over_m 	 = reshape(over_m,K,yDim*T)';
over_v   = reshape(over_v,K,yDim*T)';
y_mat = reshape(yBinary,K,yDim*T)';

varBound = -0.5*logdetAlam-0.5*WlamYSigWlamY+0.5*lam'*lam_con;                                            %prior contribution
varBound = varBound - 0.5*logdet_plds(VarInfparams.Q)*(T-1)-0.5*logdet_plds(VarInfparams.Q0);

varBound = varBound-sum(vec(feval(VarInfparams.likeHandle,y_mat,over_m,over_v,VarInfparams.dualParams)));     %likelihood contribution
%if isfield(VarInfparams,'DataBaseMeasure');   % !!! put this into PLDSInitialize or smth
%   varBound = varBound+VarInfparams.DataBaseMeasure;
%end
end
