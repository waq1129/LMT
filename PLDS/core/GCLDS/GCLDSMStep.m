function [params seq] = GCLDSMStep(params,seq)
%
% params = GCLDSMStep(params,seq) 
%


params = LDSMStepLDS(params,seq);
params = GCLDSMStepObservation(params,seq);

params = LDSTransformParams(params,'TransformType',params.opts.algorithmic.TransformType); 
