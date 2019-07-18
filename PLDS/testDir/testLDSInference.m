clear all
close all


uDim   = 3;
xDim   = 10;
yDim   = 100;
T      = 200;
Trials = 1;
Iters  = 100;

params  = LDSgenerateExample('xDim',xDim,'yDim',yDim,'uDim',uDim);
seqOrig = LDSsample(params,T,Trials);

seq = seqOrig;
[seq varBound Lambda LambdaPost] = LDSInference(params,seq);

plotPosterior(seq,1);

tic
SigFull = pinv(full(LambdaPost));
toc


VsmFull  = zeros(size(seq(1).posterior.Vsm));
VVsmFull = zeros(size(seq(1).posterior.VVsm));

for t=1:T
    xidx = (t-1)*xDim+1:t*xDim;
    VsmFull(xidx,:) = SigFull(xidx,xidx);
end

for t=1:T-1
    xidx = (t-1)*xDim+1:t*xDim;
    VVsmFull(xidx,:) = SigFull(xidx+xDim,xidx);
end

max(abs(vec(seq(1).posterior.Vsm-VsmFull)))
max(abs(vec(seq(1).posterior.VVsm-VVsmFull)))


% compare to previous code

paramsC = params.model;
paramsC.Qo = paramsC.Q0;
paramsC.xo = paramsC.x0;
seqC = seqOrig;
paramsC.notes.forceEqualT= true;
if paramsC.notes.useB;for tr=1:Trials;seqC(tr).h=seqC(tr).u;end;end
addpath('/nfs/nhome/live/lars/projects/dynamics/pair/HNLDS/matlab/PPGPFA/core_lds')
addpath('/nfs/nhome/live/lars/projects/dynamics/pair/HNLDS/matlab/PPGPFA/util')  
[seqC] = exactInferenceLDS(seqC, paramsC)

%seqC = kalmanSmootherLDS(seqC, paramsC);

figure; hold on
plot(seq(1).posterior.xsm(1,:))
plot(seqC(1).xsm(1,:),'r--')

seqC(1).Vsm = reshape(seqC(1).Vsm,xDim,xDim*T)';
seqC(1).VVsm = reshape(permute(seqC(1).VVsm(:,:,2:end),[2 1 3]),xDim,xDim*(T-1))';

figure
imagesc(seqC(1).VVsm(1:xDim,1:xDim))
figure
imagesc(seq(1).posterior.VVsm(1:xDim,1:xDim))


max(abs(vec(seq(1).posterior.Vsm-seqC(1).Vsm)))
max(abs(vec(seq(1).posterior.VVsm-seqC(1).VVsm)))

