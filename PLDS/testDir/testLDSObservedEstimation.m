clear all
close all

uDim   = 4;
xDim   = 12;
yDim   = 100;

T      = 10000;
Trials = 1;
dt     = 1;

tp  = LDSgenerateExample('xDim',xDim,'yDim',yDim,'uDim',uDim);
tp.model.Q = tp.model.Q+diag(rand(xDim,1));
%tp.model.A = diag(0.9+0.1*rand(xDim,1));
tp.model.Q0 = dlyap(tp.model.A,tp.model.Q);
seq = LDSsample(tp,T,Trials);
model.notes.useB = true;

X    = [seq.x];
Xsub = subsampleSignal([seq.x],dt);
uSub = subsampleSignal([seq.u],dt);

model = LDSObservedEstimation(Xsub,model,dt,[seq.u]);



%%% some analysis

% compare stationary covariance, very important that this is reasonable
figure
plot(vec(dlyap(tp.model.A,tp.model.Q)),vec(dlyap(model.A,model.Q)),'rx')


figure
plot(vec(tp.model.A),vec(model.A),'rx')


figure; hold on
plot(vec(tp.model.B(:,1:end-1)),vec(model.B(:,1:end-1)),'rx')
plot(vec(tp.model.B(:,end)),vec(model.B(:,end)),'bx')

figure
plot(vec(tp.model.Q),vec(model.Q),'rx')




