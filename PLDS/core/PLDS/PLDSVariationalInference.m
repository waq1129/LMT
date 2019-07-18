function [seq varBound] = PLDSVariationalInference(params,seq)
%
% [seq] = PLDSVariationalInferenceDualLDS(params,seq)
%

Trials = numel(seq);

for tr=1:Trials
  optparams.dualParams{tr} = [];
end
optparams.minFuncOptions = params.opts.algorithmic.VarInfX.minFuncOptions;

[seq] = VariationalInferenceDualLDS(params,seq,optparams);

varBound = 0;
for tr=1:Trials; varBound = varBound + seq(tr).posterior.varBound; end;
