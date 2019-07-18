function [params seq] = LDSMStep(params,seq)
%
% params = LDSMStep(params,seq) 
%


params = LDSMStepLDS(params,seq);
params = LDSMStepObservation(params,seq);

params = LDSTransformParams(params,'TransformType',params.opts.algorithmic.TransformType); 