clear all
close all


uDim   = 4;
xDim   = 12;
yDim   = 100;
T      = 200;
Trials = 1;


trueparams = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',-2.5,'uDim',uDim);
tp = trueparams;
seq = PLDSsample(trueparams,T,Trials);

Mu = getPriorMeanLDS(tp,T,'seq',seq(1));

pp = tp;
pp.model.C = zeros(size(pp.model.C));

seq = PLDSVariationalInference(pp,seq);

norm(Mu-seq(1).posterior.xsm)
