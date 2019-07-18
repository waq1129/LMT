function penalty = PLDSPenalizerPSTH(params)
%
%
%

penalty = params.opts.algorithmic.MStepNeuronPSTH.lam*0.5*norm(params.model.S,'fro').^2;