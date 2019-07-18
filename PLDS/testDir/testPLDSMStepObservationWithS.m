clear all
close all


xDim   = 3;
yDim   = 100;
T      = 150;
Trials = 10;

for tr=1:Trials
  s{tr} = (vec(repmat(rand(1,floor(T/10))>0.5,10,1))-0.5);
  s{tr} = [s{tr}' zeros(1,T-floor(T/10)*10)];
  s{tr} = repmat(s{tr},yDim,1)-1;
  s{tr}(1:20,:) = s{tr}(1:20,:)*2;
end


tp = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',0.5);
tp = LDSApplyParamsTransformation(randn(xDim)+eye(xDim)*0.3,tp);
tp.model.notes.useS = true;
seq = PLDSsample(tp,T,Trials,'s',s);
max(vec([seq.y]))

tic
seq = PLDSVariationalInference(tp,seq);
toc

% checking posterior
plotPosterior(seq,1,tp);


% do MStep
params = tp;
params.model.notes.useS = false;% true;
params = PLDSMStepObservation(params,seq);


% look at some invariant comparison statistics

subspace(tp.model.C,params.model.C)
figure
plot(vec(tp.model.C),vec(params.model.C),'xr')

figure
plot(tp.model.d,params.model.d,'xr')
