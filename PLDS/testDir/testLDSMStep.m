clear all
close all

uDim   = 3;
xDim   = 10;
yDim   = 50;
T      = 150;
Trials = 1000;

trueparams = LDSgenerateExample('xDim',xDim,'yDim',yDim,'uDim',uDim);
trueparams = LDSApplyParamsTransformation(randn(xDim)+0.1*eye(xDim),trueparams);
seq = LDSsample(trueparams,T,Trials);
seq = LDSInference(trueparams,seq);
tp  = trueparams;
tp.model.notes.learnA  = true;
tp.model.notes.learnx0 = true;


%%%%%%%%%%%%%%%%%%% test LDS Mstep %%%%%%%%%%%%%%%%%%

params = LDSMStep(tp,seq)

figure
plot(tp.model.C,params.model.C,'xr')
figure
plot(tp.model.d,params.model.d,'xr')
figure
plot(tp.model.R,params.model.R,'xr')
figure
plot(tp.model.A,params.model.A,'xr')
figure
imagesc([tp.model.A params.model.A])
figure
plot(tp.model.Q,params.model.Q,'xr')
figure
imagesc([tp.model.Q params.model.Q])
figure
plot(tp.model.Q0,params.model.Q0,'xr')
figure
imagesc([tp.model.Q0 params.model.Q0])
figure
plot(tp.model.x0,params.model.x0,'xr')

if params.model.notes.useB
  figure
  plot(tp.model.B,params.model.B,'xr')
end
