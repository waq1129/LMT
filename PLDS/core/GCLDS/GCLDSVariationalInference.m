function [seq varBound] = GCLDSVariationalInference(params,seq)
%
% [seq] = GCLDSVariationalInferenceDualLDS(params,seq)
%

%initialize by Laplace first
[seq] = GCLDSLaplaceInference(params,seq);


Trials = numel(seq);

for tr=1:Trials
  optparams.dualParams{tr} = [];
end
optparams.minFuncOptions = params.opts.algorithmic.VarInfX.minFuncOptions;

[seq] = VariationalInferenceDualGCLDS(params,seq,optparams);

varBound = 0;
for tr=1:Trials; varBound = varBound + seq(tr).posterior.varBound; end;
