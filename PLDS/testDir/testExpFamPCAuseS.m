clear all
close all

xDim   = 5;
yDim   = 100;
T      = 10000;

s = (vec(repmat(rand(1,floor(T/10))>0.5,10,1))-0.5);
s = [s' zeros(1,T-floor(T/10)*10)];
s = repmat(s,yDim,1)-1;
s(1:20,:) = s(1:20,:)*2;

tp  = PLDSgenerateExample('xDim',xDim,'yDim',yDim,'doff',0.5);
tp.model.notes.useS = true;
seq = PLDSsample(tp,T,1,'s',{s});
fprintf('Max spike count:    %i \n', max(vec([seq.y])))
fprintf('Mean spike count:   %d \n', mean(vec([seq.y])))
fprintf('Freq non-zero bin:  %d \n', mean(vec([seq.y])>0.5))

[C, X, d] = ExpFamPCA(seq.y,xDim,'s',s,'dt',5);
subspace(tp.model.C,C)   

figure
plot(tp.model.d,d,'rx')
