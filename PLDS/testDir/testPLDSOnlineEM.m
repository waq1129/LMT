clear all
close all

uDim    = 0;
xDim    = 5;
yDim    = 30;
T       = 100;
Trials  = 100;
maxIter = 100;


%%%% generate data

trueparams = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',-1.5,'uDim',uDim);
seqOrig    = PLDSsample(trueparams,T,Trials);
tp         = trueparams;
fprintf('Max spike count:    %i \n', max(vec([seqOrig.y])))
fprintf('Mean spike count:   %d \n', mean(vec([seqOrig.y])))
fprintf('Freq non-zero bin:  %d \n', mean(vec([seqOrig.y])>0.5))


%%% fit model

seq    = seqOrig;
paramsInit = PLDSInitialize(seq,xDim,'NucNormMin',[]);
fprintf('Initial subspace angle:  %d \n', subspace(tp.model.C,paramsInit.model.C))


paramsInit.model.inferenceHandle = @PLDSLaplaceInference;
%params.model.CostFuncMethod  = @PLDSVariationalInference;
%paramsInit.model.inferenceHandle = @PLDSVariationalInference;

paramsInit.opts.algorithmic.EMIterations.maxIter     = maxIter;
paramsInit.opts.algorithmic.EMIterations.maxCPUTime  = inf;
tic; [paramsOEM seqOEM varBoundOEM EStepTimesOEM MStepTimesOEM] = PopSpikeOnlineEM(paramsInit,seq); toc
fprintf('Final subspace angle:  %d \n', subspace(tp.model.C,paramsOEM.model.C))

tic; [params seq varBound EStepTimes MStepTimes] = PopSpikeEM(paramsInit,seq); toc
fprintf('Final subspace angle:  %d \n', subspace(tp.model.C,params.model.C))







%{
%%%% compare models

Tpred = 200;

%tp = LDSTransformParams(tp,'TransformType','1');
%params = LDSTransformParams(params,'TransformType','1');

seqPred = PLDSsample(trueparams,Tpred,1);

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

if params.model.notes.useB
  figure; hold on
  plot(tp.model.C*tp.model.B,params.model.C*params.model.B,'xr')
  plot(tp.model.C*tp.model.B,params.modelInit.C*params.modelInit.B,'xb')
end


figure
plot(vec(tp.model.C*tp.model.A*tp.model.Pi*tp.model.C'),vec(params.model.C*params.model.A*params.model.Pi*params.model.C'),'xr')

figure
plot(tp.model.d,params.model.d,'rx');

figure
plot(vec(tp.model.C*tp.model.Q0*tp.model.C'),vec(params.model.C*params.model.Q0*params.model.C'),'xr')

figure
plot(tp.model.C*tp.model.x0,params.model.C*params.model.x0,'xr')

%}

