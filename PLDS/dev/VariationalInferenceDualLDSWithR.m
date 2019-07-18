function [seq] = VariationalInferenceDualLDSWithR(params,seq,optparams)
%
% [seq] = VariationalInferenceDualLDSWithR(params,seq);
%
%
% do this for different normalizer than that of Poisson --> introduce base measure handle
%
%
% L Buesing 2014
%


Trials      = numel(seq);
[yDim xDim] = size(params.model.C);


% set up parameters for variational inference

VarInfparams    = params.model;
VarInfparams.CC = zeros(xDim,xDim,yDim);
for yy=1:yDim
  VarInfparams.CC(:,:,yy) = params.model.C(yy,:)'*params.model.C(yy,:);
end
VarInfparams.CC = reshape(VarInfparams.CC,xDim^2,yDim);


% iterate over trials

for tr = 1:Trials
  
  T = size(seq(tr).y,2);

  VarInfparams.d  = repmat(params.model.d,T,1);
  if params.model.notes.useS
    VarInfparams.d = VarInfparams.d + vec(seq(tr).s);
  end

  if params.model.notes.useR
    VarInfparams.R = repmat(params.model.R,T,1);
  else
    VarInfparams.R = zeros(yDim*T,1);
  end

  VarInfparams.mu = zeros(xDim,T); %prior mean
  VarInfparams.mu(:,1) = params.model.x0;
  if params.model.notes.useB;  VarInfparams.mu(:,1)=VarInfparams.mu(:,1)+params.model.B*seq(tr).u(:,1);end;
  for t=2:T; 
    VarInfparams.mu(:,t) = params.model.A*VarInfparams.mu(:,t-1); 
    if params.model.notes.useB;  VarInfparams.mu(:,t)=VarInfparams.mu(:,t)+params.model.B*seq(tr).u(:,t);end;
  end
  VarInfparams.mu = vec(VarInfparams.mu);
  
  Cl = {}; for t=1:T; Cl = {Cl{:} params.model.C}; end
  VarInfparams.W      = sparse(blkdiag(Cl{:})); %stacked loading matrix
  
  VarInfparams.y      = seq(tr).y;
  VarInfparams.Lambda = buildPriorPrecisionMatrixFromLDS(params,T);  % generate prior precision matrix
  VarInfparams.WlamW  = sparse(zeros(xDim*T)); %allocate sparse observation matrix
  % fix this: optparams.dualParams{tr} should default to 0
  VarInfparams.dualParams      = optparams.dualParams{tr};
  if isfield(params.model,'baseMeasureHandle')
    VarInfparams.DataBaseMeasure = feval(params.model.baseMeasureHandle,seq(tr).y);
  end

  % init value
  if isfield(seq(tr),'posterior')&&isfield(seq(tr).posterior,'lamInit')
    lamInit = seq(tr).posterior.lamInit;
  else
    lamInit = zeros(yDim*T,1)+mean(vec(seq(tr).y))+1e-3;
  end
  % warm start inference if possible
  if isfield(seq(tr),'posterior')&&isfield(seq(tr).posterior,'lamOpt')
    lamInit = seq(tr).posterior.lamOpt; 
  end
  
  lamOpt = minFunc(@VariationalInferenceDualCostWithR,lamInit,optparams.minFuncOptions,VarInfparams);
    
  [DualCost, ~, varBound, m_ast, invV_ast, Vsm, VVsm, over_m, over_v, n_ast, U_ast] = VariationalInferenceDualCostWithR(lamOpt,VarInfparams);


  seq(tr).posterior.xsm        = reshape(m_ast,xDim,T);	      % posterior mean   E[x(t)|y(1:T)]
  seq(tr).posterior.Vsm        = Vsm;			      % posterior covariances Cov[x(t),x(t)|y(1:T)]
  seq(tr).posterior.VVsm       = VVsm;			      % posterior covariances Cov[x(t+1),x(t)|y(1:T)]
  seq(tr).posterior.lamOpt     = lamOpt;		      % optimal value of dual variable
  seq(tr).posterior.lamInit    = lamInit;
  seq(tr).posterior.varBound   = varBound;		      % variational lower bound for trial
  seq(tr).posterior.DualCost   = DualCost;
  seq(tr).posterior.over_m     = over_m;		      % C*xsm+d
  seq(tr).posterior.over_v     = over_v;		      % diag(C*Vsm*C')

  if params.model.notes.useR
    seq(tr).posterior.n_ast    = n_ast;
    seq(tr).posterior.U_ast    = U_ast;
  end
  
end
