function params = PLDSMStepWithPSTH(params,seq)
%
% params = PLDSMStep(params,seq) 
%


params = LDSMStepLDS(params,seq);
params = PLDSMStepObservation(params,seq);
params = PLDSMStepNeuronPSTH(params,seq);

params = LDSTransformParams(params,'TransformType',params.opts.algorithmic.TransformType); 