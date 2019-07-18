function [seq] = PLDSVariationalInferenceWithR(params,seq)
%
% [seq] = PLDSVariationalInferenceDualLDSWithR(params,seq)
%

Trials = numel(seq);

for tr=1:Trials
  optparams.dualParams{tr} = [];
end
optparams.minFuncOptions = params.opts.algorithmic.VarInfX.minFuncOptions;

[seq] = VariationalInferenceDualLDSWithR(params,seq,optparams);