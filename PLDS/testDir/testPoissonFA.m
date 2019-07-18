clear all
close all

uDim    = 0;
xDim    = 5;
yDim    = 100;
T       = 100;
Trials  = 50;
maxIter = 50;


%%%% generate data

tp = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',0.5,'uDim',uDim);
tp.model.Q = dlyap(tp.model.A,tp.model.Q)/5;
tp.model.A = zeros(xDim);
tp = LDSTransformParams(tp,'TransformType','2');
tp.model.Q0 = eye(xDim);
tp.model.x0 = zeros(xDim,1);

seqOrig  = PLDSsample(tp,T,Trials);
fprintf('Max spike count:    %i \n', max(vec([seqOrig.y])))
fprintf('Mean spike count:   %d \n', mean(vec([seqOrig.y])))
fprintf('Freq non-zero bin:  %d \n', mean(vec([seqOrig.y])>0.5))


%%% fit model

seq    = seqOrig;
params = [];
if uDim>0;params.model.notes.useB = true;end
params.model.notes.learnA  = false;
params.model.notes.learnx0 = false;
params.model.notes.learnQ0 = false;
params.opts.algorithmic.ExpFamPCA.dt  = 1;
params.opts.algorithmic.TransformType = '2';


params = PLDSInitialize(seq,xDim,'ExpFamPCA',params);
fprintf('Initial subspace angle:  %d \n', subspace(tp.model.C,params.model.C))
params.model.Q  = dlyap(params.model.A,params.model.Q);
params.model.A  = zeros(xDim);
params = LDSTransformParams(params,'TransformType','2');
params.model.Q0 = eye(xDim);
params.model.x0 = zeros(xDim,1);


params.opts.algorithmic.EMIterations.maxIter     = maxIter;
params.opts.algorithmic.EMIterations.maxCPUTime  = inf;
tic; [params seq varBound EStepTimes MStepTimes] = PopSpikeEM(params,seq); toc
fprintf('Final subspace angle:  %d \n', subspace(tp.model.C,params.model.C))


%%% some plotting

subspace(tp.model.C,params.model.C)

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
plot(tp.model.d,params.model.d,'rx');
