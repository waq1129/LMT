function params = LDSMStepObservation(params,seq)
%
% function params = LDSMStepObservation(params,seq)
%

[yDim xDim] = size(params.model.C);
Trials = numel(seq);
Tall   = sum([seq.T]);

if params.model.notes.useCMask; 
  warning('params.model.notes.useCMask == true: not implemented for LDS yet')
end

%%% optimization %%%

Psi  = []; for tr=1:Trials; Psi = [Psi seq(tr).posterior.xsm]; end
Psi  = [Psi;ones(1,Tall)];

Yall = [seq.y];
if params.model.notes.useS; Yall = Yall-[seq.s]; end
YPsi = Yall*Psi';

PsiPsi = Psi*Psi';
VsmAll = zeros(xDim);
for tr=1:Trials;  VsmAll = VsmAll + sum(reshape(seq(tr).posterior.Vsm',xDim,xDim,seq(tr).T),3);end
PsiPsi(1:end-1,1:end-1) = PsiPsi(1:end-1,1:end-1) + VsmAll;

CdOpt = YPsi/(PsiPsi+eye(xDim+1)*1e-10);
params.model.C = CdOpt(:,1:end-1);
params.model.d = CdOpt(:,end);
params.model.R = cov((Yall - CdOpt*Psi)',1)+params.model.C*VsmAll*params.model.C'./sum(Tall);

if params.model.notes.diagR
  params.model.R = diag(diag(params.model.R));
end