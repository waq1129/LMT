function params = PLDSMStepObservationWithR(params,seq)
%
% function params = PLDSMStepObservation(params,seq)
%


minFuncOptions = params.opts.algorithmic.MStepObservation.minFuncOptions;

[yDim xDim] = size(params.model.C);


CdInit = vec([params.model.C params.model.d]); % warm start at current parameter values
MStepCostHandle = @PLDSMStepObservationCost;

%%% optimization %%%

CdOpt    = minFunc(MStepCostHandle,CdInit,minFuncOptions,seq,params);
CdOpt    = reshape(CdOpt,yDim,xDim+1);

params.model.C = CdOpt(:,1:xDim);
params.model.d = CdOpt(:,end);


%%% Update private variances %%%

if params.model.notes.useR && params.model.notes.learnR
  n_ast = [];
  U_ast = [];
  for tr=1:numel(seq)
    n_ast = [n_ast seq(tr).posterior.n_ast];
    U_ast = [U_ast seq(tr).posterior.U_ast];	
  end
  params.model.R = mean(U_ast+n_ast.^2,2);
end

