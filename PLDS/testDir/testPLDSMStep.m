clear all
close all


xDim   = 10;
yDim   = 100;
T      = 100;
Trials = 100;


trueparams = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',-1.5);
trueparams = LDSApplyParamsTransformation(randn(xDim)+eye(xDim)*0.3,trueparams);
seq = PLDSsample(trueparams,T,Trials);
tp  = trueparams;

tic
seq = PLDSVariationalInference(tp,seq);
toc

% checking posterior
plotPosterior(seq,1,tp);


% do MStep
params = PLDSMStep(tp,seq);


% look at some invariant comparison statistics

subspace(tp.model.C,params.model.C)

sort(eig(tp.model.A))
sort(eig(params.model.A))

tp.model.Pi     = dlyap(tp.model.A,tp.model.Q);
params.model.Pi = dlyap(params.model.A,params.model.Q);

figure
plot(vec(tp.model.C*tp.model.Pi*tp.model.C'),vec(params.model.C*params.model.Pi*params.model.C'),'xr')

figure
plot(vec(tp.model.C*tp.model.A*tp.model.Pi*tp.model.C'),vec(params.model.C*params.model.A*params.model.Pi*params.model.C'),'xr')

figure
plot(tp.model.d,params.model.d,'rx');

figure
plot(vec(tp.model.C*tp.model.Q0*tp.model.C'),vec(params.model.C*params.model.Q0*params.model.C'),'xr')

figure
plot(tp.model.C*tp.model.x0,params.model.C*params.model.x0,'xr')

