clear all
close all

uDim    = 0;
xDim    = 3;
yDim    = 30;
T       = 200;
Trials  = 1;


%%%% ground truth model

for tr=1:Trials
  s{tr} = (vec(repmat(rand(1,floor(T(tr)/10))>0.5,10,1))-0.5);
  s{tr} = [s{tr}' zeros(1,T(tr)-floor(T(tr)/10)*10)];
  s{tr} = repmat(s{tr},yDim,1);
%  s{tr}(1:20,:) = s{tr}(1:20,:)*2;
  s{tr} = s{tr}*4;
end

tp = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',-2.5,'uDim',uDim);
tp.model.notes.useS = true;

UA = randn(xDim); UA = (UA-UA')/2;
tp.model.A = expm(0.1*UA)*0.995*eye(xDim);
eig(tp.model.A)

seqOrig = PLDSsample(tp,T,Trials,'s',s);
seq     = seqOrig;
params  = tp;


%%%% prediction example

y = seq(1).y;
max(vec(y))

condRange = [50:125];
predRange = [126:190];

tic; [ypred xpred xpredCov seqInf] = PLDSPredictRange(params,y,condRange,predRange,'s',s{1}); toc

figure
imagesc([y(:,predRange) ypred])


figure; hold on
plot(seq(1).x','k')
plot(condRange,seqInf(1).posterior.xsm','b')
plot(predRange,xpred','r')

