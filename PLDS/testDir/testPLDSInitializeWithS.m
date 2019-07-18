clear all
close all


uDim   = 0;
xDim   = 12;
yDim   = 100;
T      = 200;
Trials = 200;

for tr=1:Trials
  s{tr} = (vec(repmat(rand(1,floor(T/10))>0.5,10,1))-0.5);
  s{tr} = [s{tr}' zeros(1,T-floor(T/10)*10)];
  s{tr} = repmat(s{tr},yDim,1)-1;
  s{tr}(1:20,:) = s{tr}(1:20,:)*2;
end

tp = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',-11.0,'uDim',uDim);
tp.model.notes.useS = true;
seq = PLDSsample(tp,T,Trials,'s',s);
mean(vec([seq.y]))


params = [];
params.model.notes.useS = true;
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

if uDim>0
  figure
  plot(vec(tp.model.C*tp.model.B),vec(params.model.C*params.model.B),'xr')
end


