clear all
close all


xDim   = 10;
yDim   = 100;
T      = 100;
Trials = 150;


trueparams = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',-1.5);
trueparams = LDSApplyParamsTransformation(randn(xDim)+eye(xDim)*0.3,trueparams);
seq = PLDSsample(trueparams,T,Trials);
tp =  trueparams;

tic
seq = PLDSVariationalInference(tp,seq);
toc

% checking posterior
plotPosterior(seq,1,tp);


% do MStep
params = LDSMStepLDS(tp,seq);


% look at some invariant comparison statistics


sort(eig(tp.model.A))
sort(eig(params.model.A))

tp.model.Pi     = dlyap(tp.model.A,tp.model.Q);
params.model.Pi = dlyap(params.model.A,params.model.Q);

figure
plot(vec(tp.model.A),vec(params.model.A),'xr')

figure
plot(vec(tp.model.Q),vec(params.model.Q),'xr')

figure
plot(vec(tp.model.Pi),vec(params.model.Pi),'xr')

figure
plot(vec(tp.model.x0),vec(params.model.x0),'xr')

figure
plot(vec(tp.model.Q0),vec(params.model.Q0),'xr')

