clear all
close all

uDim   = 0;
xDim   = 5;
yDim   = 50;
T      = 100;
Trials = 1;


params = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',-1.5);%,'uDim',uDim);
params.model.notes.useR = true;
params.model.R = 0.3*rand(yDim,1)+0.1;
seq = PLDSsample(params,T,Trials);
max(vec([seq.y]))


tic
seqnR = PLDSVariationalInference(params,seq);
toc

%params.model.R = zeros(yDim,1);
tic
seqR = PLDSVariationalInferenceWithR(params,seq);
toc


plotPosterior(seqnR,1,params);
plotPosterior(seqR,1,params);

norm([seqnR(1).posterior.xsm]-[seqR(1).posterior.xsm],'fro')
seqnR(1).posterior.varBound
seqR(1).posterior.varBound



if uDim>0
  mu = zeros(xDim,T);
  mu(:,1) = params.model.B*seq(1).u(:,1);
  for t=2:T
    mu(:,t) = params.model.A*mu(:,t-1)+params.model.B*seq(1).u(:,t);
  end
  figure
  plot(mu')
end
