function [params seq] = PLDSLinkFuncMStep(params,seq)
%
% params = PLDSLinkFuncMStep(params,seq)
%


params = LDSMStepLDS(params,seq);
params = PLDSLinkFuncMStepObservation(params,seq);

params = LDSTransformParams(params,'TransformType',params.opts.algorithmic.TransformType); 


% experimental feature: average, smoothed substract posterior mean
if params.model.notes.useS && params.opts.algorithmic.MStep.subPostMean% && (params.opts.EMiter>20)
  xDim = size(params.model.A,1);
  [songs Trials] = size(seq);
  %disp('sub mean posterior')
  for ss=1:songs
    mxsm = [];
    for tr=1:Trials; mxsm = [mxsm seq(ss,tr).posterior.xsm]; end
    mxsm = smoothTrajectory(mean(reshape(mxsm,xDim,seq(ss,1).T,Trials),3),params.opts.algorithmic.MStep.smoothPostMean);
    for tr=1:Trials
      seq(ss,tr).s = seq(ss,tr).s+params.model.C*mxsm;
    end
  end
end
