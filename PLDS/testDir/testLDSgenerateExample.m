clear all
close all

uDim   = 0;
xDim   = 12;
yDim   = 100;

T      = 200;
Trials = 2;


tp  = LDSgenerateExample('xDim',xDim,'yDim',yDim,'uDim',uDim);
seq = LDSsample(tp,T,Trials);

if tp.model.notes.useB
  figure
  plot(seq(1).u')
end  

figure
plot(seq(1).x')

figure
imagesc(seq(1).y)