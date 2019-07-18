clear all
close all


xDim   = 3;
yDim   = 100;
T      = 100;
Trials = 500;


%%%% generate data

trueparams = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',0.0);
seqOrig    = PLDSsample(trueparams,T,Trials);
tp         = trueparams;


%%% initialize with exponential family PCA

seq    = seqOrig;
params = [];
params = PLDSInitialize(seq,xDim,'ExpFamPCA',params);
subspace(tp.model.C,params.model.C)

%%% do 10 EM iterations
%
%params.startParams = params;
%params.opts.algorithmic.EMIterations.maxIter = 10;
%[params varBound] = PopSpikeEM(params,seq);
%subspace(tp.model.C,params.model.C)

%%% plot some diagnostics


sort(eig(tp.model.A))
sort(eig(params.model.A))


tp.model.Pi     = dlyap(tp.model.A,tp.model.Q);
params.model.Pi = dlyap(params.model.A,params.model.Q);

figure
plot(vec(tp.model.C*tp.model.Pi*tp.model.C'),vec(params.model.C*params.model.Pi*params.model.C'),'xr')

%{
figure
plot(vec(tp.model.C*tp.model.A*tp.model.Pi*tp.model.C'),vec(params.model.C*params.model.A*params.model.Pi*params.model.C'),'xr')

figure
plot(tp.model.d,params.model.d,'rx');

%}