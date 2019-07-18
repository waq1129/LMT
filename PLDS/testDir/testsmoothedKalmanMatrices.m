clear all
close all


xDim   = 20;
yDim   = 100;
T      = 100;
Trials = 3;
Iters  = 100;

params = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim);

Lambda = buildPriorPrecisionMatrixFromLDS(params,T);

Rinv = randn(yDim); Rinv = Rinv*Rinv';
Rinv = Rinv./max(abs(eig(Rinv)));

CRinvCFull = zeros(T*xDim,T*xDim);
CRinvC     = zeros(T*xDim,xDim);
crinvc     = params.C'*Rinv*params.C;
for t=1:T
    xidx = (t-1)*xDim+1:t*xDim;
    CRinvCFull(xidx,xidx) = crinvc;
    CRinvC(xidx,:) = crinvc;
end


tic
SigFull = pinv(Lambda+CRinvCFull);
toc

tic
[Vsm VVsm] = smoothedKalmanMatrices(params,CRinvC);
toc

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


