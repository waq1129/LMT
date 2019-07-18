clear all
close all

xDim   = 10;
yDim   = 100;

T      = [100 50 120];
Trials = numel(T);

for tr=1:Trials
  s{tr} = (vec(repmat(rand(1,floor(T(tr)/10))>0.5,10,1))-0.5);
  s{tr} = [s{tr}' zeros(1,T(tr)-floor(T(tr)/10)*10)];
  s{tr} = repmat(s{tr},yDim,1);
  s{tr}(1:20,:) = s{tr}(1:20,:)*2;
end

params = PLDSgenerateExample('xDim',xDim,'yDim',yDim);
params.model.notes.useS = true;
seq    = PLDSsample(params,T,Trials,'s',s);

mean(vec([seq.y]))
figure
plot(seq(1).x')
figure
imagesc(seq(1).y)