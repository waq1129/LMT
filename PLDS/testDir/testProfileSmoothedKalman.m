clear all
close all


xDim   = 100;
yDim   = 100;
T      = 10000;
Trials = 3;
Iters  = 100;

params = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim);

%Lambda = buildPriorPrecisionMatrixFromLDS(params,T);

Rinv = randn(yDim); Rinv = Rinv*Rinv';
Rinv = Rinv./max(abs(eig(Rinv)));

%CRinvCFull = zeros(T*xDim,T*xDim);
CRinvC     = zeros(T*xDim,xDim);
crinvc     = params.model.C'*Rinv*params.model.C;
for t=1:T
    xidx = (t-1)*xDim+1:t*xDim;
%    CRinvCFull(xidx,xidx) = crinvc;
    CRinvC(xidx,:) = crinvc;
end

%
%tic
%SigFull = pinv(Lambda+CRinvCFull);
%toc

tic
[Vsm VVsm] = smoothedKalmanMatrices(params,CRinvC);
toc

for t=1:T
  xidx = (t-1)*xDim+1:t*xDim;
  AA(:,:,t) = params.model.A*params.model.Q*params.model.A'+CRinvC(xidx,:);
end
%Lambda = full(Lambda);
for t=1:T-1
  xidx = (t-1)*xDim+1:t*xDim;
  BB(:,:,t) = - params.model.A*params.model.Q;%Lambda(xidx+xDim,xidx); 
end

tic
[Ve VVe] =  sym_blk_tridiag_inv_v1(AA, BB, [1:T]', ones(T-1,1));
toc

%{
VsmFull  = zeros(size(Vsm));
VVsmFull = zeros(size(VVsm));

for t=1:T
    xidx = (t-1)*xDim+1:t*xDim;
    VsmFull(xidx,:) = SigFull(xidx,xidx);
end

for t=1:T-1
    xidx = (t-1)*xDim+1:t*xDim;
    VVsmFull(xidx,:) = SigFull(xidx+xDim,xidx);
end

max(abs(vec(Vsm-VsmFull)))
max(abs(vec(VVsm-VVsmFull)))


%}