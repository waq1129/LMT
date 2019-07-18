function params = GCLDSMStepObservation(params,seq)
%
% function params = GCLDSMStepObservation(params,seq)
%
% Yuanjun Gao, 2015


minFuncOptions = params.opts.algorithmic.MStepObservation.minFuncOptions;

[yDim, xDim] = size(params.model.C);
K = size(params.model.g, 2);

if params.model.notes.useCMask; params.model.C = params.model.C.*params.model.CMask; end

%different structure on g
if params.model.notes.gStatus == 2, %share same curvature but different slope
    d = params.model.g(:,1);
    g_adjusted = params.model.g - bsxfun(@times, d, 1:K);
    g_adjusted = mean(g_adjusted(:,2:end),1);
    CgInit = [vec(params.model.C); vec(d); vec(g_adjusted)];
elseif params.model.notes.gStatus == 1, %share everything
    CgInit = [vec(params.model.C); vec(mean(params.model.g))];
elseif params.model.notes.gStatus == 0, %different curvature
    CgInit = vec([params.model.C, params.model.g]); % warm start at current parameter values
elseif params.model.notes.gStatus == 3,     %also truncated PLDS but truncated at the same value for all neurons
    CgInit = vec([params.model.C, params.model.g(:,1)]); % warm start at current parameter values
end
    
MStepCostHandle = @GCLDSMStepObservationCost;

%%% optimization %%%
CgOpt = minFunc(MStepCostHandle,CgInit,minFuncOptions,seq,params);

%get back to the parameter
if params.model.notes.gStatus == 2,
    params.model.C = reshape(CgOpt(1:(yDim * xDim)), yDim, xDim);
    d = reshape(CgOpt((yDim*xDim+1):(yDim*(xDim+1))), yDim, 1);
    g_adjusted = reshape(CgOpt((yDim*(xDim+1)+1):end), 1, []);
    params.model.g = bsxfun(@plus, bsxfun(@times, d, 1:K), [0,g_adjusted]);
elseif params.model.notes.gStatus == 1,
    params.model.C = reshape(CgOpt(1:(yDim * xDim)), yDim, xDim);
    params.model.g = repmat(reshape(CgOpt((yDim*xDim+1):end), 1, K), yDim, 1);
elseif params.model.notes.gStatus == 0,
    CgOpt = reshape(CgOpt,yDim,xDim+K);
    params.model.C = CgOpt(:,1:xDim);
    params.model.g = CgOpt(:,(xDim+1):end);
elseif params.model.notes.gStatus == 3,
    CgOpt = reshape(CgOpt,yDim,xDim+1);
    params.model.C = CgOpt(:,1:xDim);
    params.model.g = bsxfun(@times, CgOpt(:,end), 1:K);
end

if params.model.notes.useCMask; params.model.C = params.model.C.*params.model.CMask; end


