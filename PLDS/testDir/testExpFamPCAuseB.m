clear all
close all

uDim   = 2;
xDim   = 5;
yDim   = 100;

T      = 100;
Trials = 100;

tp  = PLDSgenerateExample('xDim',xDim,'yDim',yDim,'uDim',uDim);
seq = PLDSsample(tp,T,Trials);

params.model.notes.useB = true;
%params = PLDSInitialize(seq,xDim,'ExpFamPCA',params)
params = PLDSInitialize(seq,xDim,'NucNormMin',params)

subspace(tp.model.C,params.model.C)

figure
plot(tp.model.d,params.model.d,'rx')

figure
tp.model.Pi = dlyap(tp.model.A,tp.model.Q);
params.model.Pi =  dlyap(params.model.A,params.model.Q);
plot(vec(tp.model.C*tp.model.Pi*tp.model.C'),vec(params.model.C*params.model.Pi*params.model.C'),'rx');

figure
plot(vec(tp.model.C*tp.model.B),vec(params.model.C*params.model.B),'rx')




%{

%mean(vec([seq.y]))
%figure
%plot(seq(1).x')
%figure
%imagesc(seq(1).y)

%options.display     = 'none';
options.MaxIter     = 10000;
options.maxFunEvals = 50000;
options.Method      = 'lbfgs';
options.progTol     = 1e-9;
options.optTol      = 1e-5;


[Cest, Xest, dest] = ExpFamPCA([seq.y],xDim,'options',options,'dt',10,'lam',0.1);
subspace(tp.model.C,Cest) 

figure
plot(tp.model.d,dest,'rx')

figure
plot(vec(tp.model.C*dlyap(tp.model.A,tp.model.Q)*tp.model.C'),vec(Cest*cov(Xest')*Cest'),'rx');

Cest'*Cest
Xest*Xest'./T

[UC SC VC] = svd(Cest);
M = SC(1:xDim,1:xDim)*VC(:,1:xDim)';
Cest = Cest/M;
Xest = M*Xest;
U = C'*Cest;
Cest = Cest/U;
Xest = U*Xest;


figure
plot(vec(X),vec(Xest),'rx');

figure
plot(vec(d),vec(dest),'rx'); 


%}