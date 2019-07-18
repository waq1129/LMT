clear all
close all


uDim   = 4;
xDim   = 12;
yDim   = 100;
T      = 200;
Trials = 20;


trueparams = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',-2.5,'uDim',uDim);
seq = PLDSsample(trueparams,T,Trials);
mean(vec([seq.y]))
tp  = trueparams;

params = [];
params.model.notes.useB = (uDim>0.5);
params = PLDSInitialize(seq,xDim,'NucNormMin',params);
%params = PLDSInitialize(seq,xDim,'ExpFamPCA',params);


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

%figure
%plot(vec(tp.model.C*tp.model.Q0*tp.model.C'),vec(params.model.C*params.model.Q0*params.model.C'),'xr')

figure
plot(vec(tp.model.C*tp.model.B),vec(params.model.C*params.model.B),'xr')

%figure
%plot(tp.model.C*tp.model.x0,params.model.C*params.model.x0,'xr')

