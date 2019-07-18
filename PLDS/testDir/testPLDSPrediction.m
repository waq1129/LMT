clear all
close all

uDim    = 2;
xDim    = 3;
yDim    = 30;
T       = 200;
Trials  = 1;


%%%% ground truth model

tp = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',-2.0,'uDim',uDim);
tp.model.B = tp.model.B;
tp.model.Q = tp.model.Q/10;

UA = randn(xDim); UA = (UA-UA')/2;
tp.model.A       = expm(0.1*UA)*0.995*eye(xDim);
eig(tp.model.A)

seqOrig = PLDSsample(tp,T,Trials);
seq     = seqOrig;
params  = tp;


%%%% prediction example

y = seq(1).y;
max(vec(y))

condRange = [50:125];
predRange = [126:190];

tic; [ypred xpred xpredCov seqInf] = PLDSPredictRange(params,y,condRange,predRange,'u',seq(1).u); toc

figure
imagesc([y(:,predRange) ypred])


figure; hold on
plot(seq(1).x','k')
plot(condRange,seqInf(1).posterior.xsm','b')
plot(predRange,xpred','r')

