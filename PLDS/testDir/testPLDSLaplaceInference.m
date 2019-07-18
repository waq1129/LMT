clear all
close all

%rng('default');

uDim   = 0;   
xDim   = 3;
yDim   = 30;
T      = 100;
Trials = 3;


params  = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'uDim',uDim,'doff',-1.75);
seqOrig = PLDSsample(params,T,Trials);


tic
[seqVarInf varBoundVB] = PLDSVariationalInference(params,seqOrig);
toc

tic
[seqLpInf varBoundLP] = PLDSLaplaceInference(params,seqOrig);
toc

Mu = getPriorMeanLDS(params,T,'seq',seqOrig(1));
norm(seqVarInf(1).posterior.xsm-seqLpInf(1).posterior.xsm,'fro')/norm(seqVarInf(1).posterior.xsm-Mu,'fro')



figure;
plot(vec(seqVarInf.posterior.Vsm),vec(seqLpInf.posterior.Vsm),'rx')

figure;
plot(vec(seqVarInf.posterior.xsm),vec(seqLpInf.posterior.xsm),'rx')

figure;
plot(vec(seqVarInf.posterior.VVsm),vec(seqLpInf.posterior.VVsm),'rx')

