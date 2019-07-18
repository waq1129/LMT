function params = PLDSMStepObservation(params,seq)
%
% function params = PLDSMStepObservation(params,seq)
%


minFuncOptions = params.opts.algorithmic.MStepObservation.minFuncOptions;

[yDim xDim] = size(params.model.C);

if params.model.notes.useCMask; params.model.C = params.model.C.*params.model.CMask; end
CdInit = vec([params.model.C params.model.d]); % warm start at current parameter values
MStepCostHandle = @PLDSMStepObservationCost;

%%% optimization %%%

CdOpt = minFunc(MStepCostHandle,CdInit,minFuncOptions,seq,params);
CdOpt = reshape(CdOpt,yDim,xDim+1);

params.model.C = CdOpt(:,1:xDim);
if params.model.notes.useCMask; params.model.C = params.model.C.*params.model.CMask; end
params.model.d = CdOpt(:,end);


