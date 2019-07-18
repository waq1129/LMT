clear all
close all

uDim    = 2;
xDim    = 3;
yDim    = 50;
T       = 100;
Trials  = 250;
maxIter = 25;


%%%% generate data


for tr=1:Trials
  s{tr} = (vec(repmat(rand(1,floor(T/10))>0.5,10,1))-0.5);
  s{tr} = [s{tr}' zeros(1,T-floor(T/10)*10)];
  s{tr} = repmat(s{tr},yDim,1)-1;
  s{tr}(1:20,:) = s{tr}(1:20,:)*1.5;
end

tp = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',-1.0);
tp.model.notes.useS = true;
seqOrig    = PLDSsample(tp,T,Trials,'s',s);
tp         = tp;
fprintf('Max spike count:    %i \n', max(vec([seqOrig.y])))
fprintf('Mean spike count:   %d \n', mean(vec([seqOrig.y])))
fprintf('Freq non-zero bin:  %d \n', mean(vec([seqOrig.y])>0.5))


%%% fit model

seq    = seqOrig;
params = [];
%params.model.notes.useS = true;
params = PLDSInitialize(seq,xDim,'ExpFamPCA',params);
%params = PLDSInitialize(seq,xDim,'NucNormMin',params);
fprintf('Initial subspace angle:  %d \n', subspace(tp.model.C,params.model.C))

params.model.inferenceHandle = @PLDSLaplaceInference;
%params.model.inferenceHandle = @PLDSVariationalInference;
params.opts.algorithmic.EMIterations.maxIter     = maxIter;
params.opts.algorithmic.EMIterations.maxCPUTime  = inf;
tic; [params seq varBound EStepTimes MStepTimes] = PopSpikeEM(params,seq); toc
fprintf('Final subspace angle:  %d \n', subspace(tp.model.C,params.model.C))


figure
plot(tp.model.d,params.model.d,'rx')

tp.model.Pi     = dlyap(tp.model.A,tp.model.Q);
params.model.Pi = dlyap(params.model.A,params.model.Q);

figure
plot(vec(tp.model.C*tp.model.Pi*tp.model.C'),vec(params.model.C*params.model.Pi*params.model.C'),'xr')

figure
plot(vec(tp.model.C*tp.model.A*tp.model.Pi*tp.model.C'),vec(params.model.C*params.model.A*params.model.Pi*params.model.C'),'xr')




%{
%%%% compare models

Tpred = 200;

%tp = LDSTransformParams(tp,'TransformType','1');
%params = LDSTransformParams(params,'TransformType','1');

seqPred = PLDSsample(tp,Tpred,1);

condRange = [1:100];
predRange = [101:200];

tic; [ypred xpred xpredCov seqInf] = PLDSPredictRange(params,seqPred(1).y,condRange,predRange,'u',seqPred(1).u); toc
tic; [ypredTP xpredTP xpredCovTP seqInfTP] = PLDSPredictRange(tp,seqPred(1).y,condRange,predRange,'u',seqPred(1).u); toc

figure
imagesc([seqPred.y(:,predRange) ypred])

figure; hold on
plot(vec([seqPred.y(:,predRange)])+0.5-rand(numel([seqPred.y(:,predRange)]),1),vec(ypred),'rx')

figure; hold on
plot(vec(ypredTP),vec(ypred),'rx')


figure; hold on
plot(seqPred(1).x','k')
plot(condRange,seqInfTP(1).posterior.xsm','b')
plot(predRange,xpredTP','r')





%%% some plotting

subspace(tp.model.C,params.model.C)

sort(eig(tp.model.A))
sort(eig(params.model.A))

tp.model.Pi     = dlyap(tp.model.A,tp.model.Q);
params.model.Pi = dlyap(params.model.A,params.model.Q);

figure
plot(vec(tp.model.C*tp.model.Pi*tp.model.C'),vec(params.model.C*params.model.Pi*params.model.C'),'xr')

if params.model.useB
  figure; hold on
  plot(tp.model.C*tp.model.B,params.model.C*params.model.B,'xr')
  plot(tp.model.C*tp.model.B,params.modelInit.C*params.modelInit.B,'xb')
end

save('/nfs/data3/lars/dynamics/popspikedyn/long_run.mat')



figure
plot(vec(tp.model.C*tp.model.A*tp.model.Pi*tp.model.C'),vec(params.model.C*params.model.A*params.model.Pi*params.model.C'),'xr')

figure
plot(tp.model.d,params.model.d,'rx');

figure
plot(vec(tp.model.C*tp.model.Q0*tp.model.C'),vec(params.model.C*params.model.Q0*params.model.C'),'xr')

figure
plot(tp.model.C*tp.model.x0,params.model.C*params.model.x0,'xr')

%}