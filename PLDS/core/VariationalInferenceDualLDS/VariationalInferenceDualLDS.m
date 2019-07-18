function [seq] = VariationalInferenceDualLDS(params,seq,optparams)
%
% [seq] = VariationalInferenceDualLDS(params,seq);
%
%
% L Buesing 2014
%


Trials      = numel(seq);
[yDim xDim] = size(params.model.C);
Tmax        = max([seq.T]);

% set up parameters for variational inference

VarInfparams    = params.model;
VarInfparams.CC = zeros(xDim,xDim,yDim);
for yy=1:yDim
  VarInfparams.CC(:,:,yy) = params.model.C(yy,:)'*params.model.C(yy,:);
end
VarInfparams.CC = reshape(VarInfparams.CC,xDim^2,yDim);

Cl = {}; for t=1:Tmax; Cl = {Cl{:} params.model.C}; end
Wmax = sparse(blkdiag(Cl{:}));

% iterate over trials

for tr = 1:Trials
  
  T = size(seq(tr).y,2);

  VarInfparams.d          = repmat(params.model.d,T,1); if params.model.notes.useS; VarInfparams.d = VarInfparams.d + vec(seq(tr).s); end
  VarInfparams.mu         = vec(getPriorMeanLDS(params,T,'seq',seq(tr)));
  VarInfparams.W          = Wmax(1:yDim*T,1:xDim*T);  
  VarInfparams.y          = seq(tr).y;
  VarInfparams.Lambda     = buildPriorPrecisionMatrixFromLDS(params,T);  % generate prior precision matrix
  VarInfparams.WlamW      = sparse(zeros(xDim*T)); %allocate sparse observation matrix
  VarInfparams.dualParams = optparams.dualParams{tr};
  
  if isfield(params.model,'baseMeasureHandle')
    VarInfparams.DataBaseMeasure = feval(params.model.baseMeasureHandle,seq(tr).y,params);
    seq(tr).posterior.DataBaseMeasure = VarInfparams.DataBaseMeasure;
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
  
  
  lamOpt = minFunc(@VariationalInferenceDualCost,lamInit,optparams.minFuncOptions,VarInfparams);
    
  [DualCost, ~, varBound, m_ast, invV_ast, Vsm, VVsm, over_m, over_v] = VariationalInferenceDualCost(lamOpt,VarInfparams);


  seq(tr).posterior.xsm        = reshape(m_ast,xDim,T);	      % posterior mean   E[x(t)|y(1:T)]
  seq(tr).posterior.Vsm        = Vsm;			      % posterior covariances Cov[x(t),x(t)|y(1:T)]
  seq(tr).posterior.VVsm       = VVsm;			      % posterior covariances Cov[x(t+1),x(t)|y(1:T)]
  seq(tr).posterior.lamOpt     = lamOpt;		      % optimal value of dual variable
  seq(tr).posterior.lamInit    = lamInit;
  seq(tr).posterior.varBound   = varBound;		      % variational lower bound for trial
  seq(tr).posterior.DualCost   = DualCost;
  seq(tr).posterior.over_m     = over_m;		      % C*xsm+d
  seq(tr).posterior.over_v     = over_v;		      % diag(C*Vsm*C')
  
end
