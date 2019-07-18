clear all
close all

uDim   = 0;
xDim   = 10;
yDim   = 100;
T      = [100];
Trials = 1;

for tr=1:Trials
  s{tr} = (vec(repmat(rand(1,floor(T(tr)/10))>0.5,10,1))-0.5);
  s{tr} = [s{tr}' zeros(1,T(tr)-floor(T(tr)/10)*10)];
  s{tr} = repmat(s{tr},yDim,1);
  s{tr}(1:20,:) = s{tr}(1:20,:);
  s{tr} = s{tr}*5;
end


params = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',-2.5,'uDim',uDim);
params.model.notes.useS = true;
seq = PLDSsample(params,T,Trials,'s',s);
max(vec([seq.y]))

seqInf = seq;
tic
seqInf = PLDSVariationalInference(params,seqInf);
toc

seqInfLp = seq;
tic
seqInfLp = PLDSlpinf(params,seqInfLp);
toc


plotPosterior(seqInf,1,params);
plotPosterior(seqInfLp,1,params);


