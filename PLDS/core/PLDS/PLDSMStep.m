function [params seq] = PLDSMStep(params,seq)
%
% params = PLDSMStep(params,seq) 
%


params = LDSMStepLDS(params,seq);
params = PLDSMStepObservation(params,seq);

params = LDSTransformParams(params,'TransformType',params.opts.algorithmic.TransformType); 