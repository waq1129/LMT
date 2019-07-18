% PLDS toolbox example, with external input
%
% Lars Buesing, Jakob H Macke, 2014
%


clear all
close all

uDim    = 2;     % dimension of external input to latent states
xDim    = 3;
yDim    = 30;
T       = 100;
Trials  = 50;
maxIter = 100;


%%%% generate data

trueparams = PLDSgenerateExample('xDim',xDim,'yDim',yDim,'doff',-1.5,'uDim',uDim);
seqOrig    = PLDSsample(trueparams,T,Trials,'yMax',10);
tp         = trueparams;
fprintf('Max spike count:    %i \n', max(vec([seqOrig.y])))
fprintf('Mean spike count:   %d \n', mean(vec([seqOrig.y])))
fprintf('Freq non-zero bin:  %d \n', mean(vec([seqOrig.y])>0.5))


%%% fit model

seq    = seqOrig;
params = [];
% important: set flag to use external input
if uDim>0;params.model.notes.useB = true;end


params = PLDSInitialize(seq,xDim,'NucNormMin',params);
fprintf('Initial subspace angle:  %d \n', subspace(tp.model.C,params.model.C))


params.model.inferenceHandle = @PLDSLaplaceInference;
params.opts.algorithmic.EMIterations.maxIter     = maxIter;
params.opts.algorithmic.EMIterations.maxCPUTime  = inf;
tic; [params seq varBound EStepTimes MStepTimes] = PopSpikeEM(params,seq); toc
fprintf('Final subspace angle:  %d \n', subspace(tp.model.C,params.model.C))


%%% check learned parameters

figure
plot(vec(tp.model.C*tp.model.B),vec(params.model.C*params.model.B),'xr')


%%% test model by predicting future spike trains 

Tpred = 200;
seqPred = PLDSsample(trueparams,Tpred,1);  % sample a test data set

condRange = [1:100];                       % the time interval to condition on
predRange = [101:200];                     % the time interval to predict

% predict with learned parameters
tic; [ypred xpred xpredCov seqInf] = PLDSPredictRange(params,seqPred(1).y,condRange,predRange,'u',seqPred(1).u); toc


figure;
subplot(2,1,1)
imagesc(seqPred.y(:,predRange))
title('true data')
subplot(2,1,2)
imagesc(ypred)
title('prediction')
