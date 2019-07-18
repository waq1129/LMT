function [seq varBound] = PLDSLaplaceInference(params,seq)
%
% function seq = PLDSlpinf(params,seq)
%

computeVarBound = true; %!!! make this nice

Trials      = numel(seq);
[yDim xDim] = size(params.model.C);
varBound    = 0;

mps = params.model;
mps.E = [];
mps.D = [];
mps.initV = mps.Q0;
mps.initx = mps.x0;

runinfo.nStateDim = xDim;
runinfo.nObsDim   = yDim;

infparams.initparams = mps;
infparams.runinfo    = runinfo;
infparams.notes      = params.model.notes;

Tmax = max([seq.T]);
for tr=1:Trials
    T = size(seq(tr).y,2);
    indat = seq(tr);
    if isfield(seq(tr),'posterior') && isfield(seq(tr).posterior,'xsm')
        indat.xxInit = seq(tr).posterior.xsm;
    else
        indat.xxInit = getPriorMeanLDS(params,T,'seq',seq(tr));
    end
    seqRet = PLDSLaplaceInferenceCore(indat, infparams);
    seq(tr).posterior.xsm      = seqRet.x;
    seq(tr).posterior.Vsm      = reshape(seqRet.V,xDim,xDim*T)';
    seq(tr).posterior.VVsm     = reshape(permute(seqRet.VV(:,:,2:end),[2 1 3]),xDim,xDim*(T-1))';
    seq(tr).posterior.lamOpt   = exp(vec(seqRet.Ypred));
end


if computeVarBound
    
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
        VarInfparams.d = repmat(params.model.d,T,1); if params.model.notes.useS; VarInfparams.d = VarInfparams.d + vec(seq(tr).s); end
        VarInfparams.mu         = vec(getPriorMeanLDS(params,T,'seq',seq(tr)));
        VarInfparams.W          = Wmax(1:yDim*T,1:xDim*T);
        VarInfparams.y          = seq(tr).y;
        VarInfparams.Lambda     = buildPriorPrecisionMatrixFromLDS(params,T);
        VarInfparams.WlamW      = sparse(zeros(xDim*T));
        VarInfparams.dualParams = [];
        
        if isfield(params.model,'baseMeasureHandle')
            VarInfparams.DataBaseMeasure = feval(params.model.baseMeasureHandle,seq(tr).y,params);
            seq(tr).posterior.DataBaseMeasure = VarInfparams.DataBaseMeasure;
        end
        
        lamOpt = seq(tr).posterior.lamOpt;
        [DualCost, ~, varBound] = VariationalInferenceDualCost(lamOpt,VarInfparams);
        seq(tr).posterior.varBound = varBound;
        
    end
    varBound = 0;
    for tr=1:Trials; varBound = varBound + seq(tr).posterior.varBound; end;
end

