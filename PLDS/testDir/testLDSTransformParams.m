clear all
close all


uDim   = 0
xDim   = 10;
yDim   = 100;
T      = 1000;
Trials = 1;    


trueparams = LDSgenerateExample('xDim',xDim,'yDim',yDim,'uDim',uDim);
seqOrig    = LDSsample(trueparams,T,Trials);

tp = trueparams;
tp.model.PiY = tp.model.C*tp.model.Pi*tp.model.C';


%%%%%%%%% test parameter transformation

% random transformation works!

[params] = LDSApplyParamsTransformation(randn(xDim)+0.1*eye(xDim),tp);
[params] = LDSTransformParams(params,'TransformType','5');

params.model.C'*params.model.C
dlyap(params.model.A,params.model.Q)


params.model.PiY = params.model.C*params.model.Pi*params.model.C';

figure
plot(vec(tp.model.PiY),vec(params.model.PiY),'rx');

figure    
plot(vec(tp.model.C*tp.model.A*tp.model.Pi*tp.model.C'),vec(params.model.C*params.model.A*params.model.Pi*params.model.C'),'rx');

figure
plot(vec(tp.model.C*tp.model.Q0*tp.model.C'),vec(params.model.C*params.model.Q0*params.model.C'),'rx');

figure
plot(vec(tp.model.C*tp.model.x0),vec(params.model.C*params.model.x0),'rx');

if params.model.notes.useB
  figure
  plot(vec(tp.model.C*tp.model.B),vec(params.model.C*params.model.B),'rx');
end


subspace(tp.model.C,params.model.C)

sort(eig(tp.model.A))
sort(eig(params.model.A))




%{

%% this needs to be debugged, not really vital

tp.notes.forceEqualT = true;
tp.notes.useB = false;
tp.notes.useD = false;
tp.notes.type = 'LDS';

addpath('/nfs/nhome/live/lars/projects/dynamics/pair/HNLDS/matlab/PPGPFA/core_lds')
addpath('/nfs/nhome/live/lars/projects/dynamics/pair/HNLDS/matlab/PPGPFA/util')

tp.xo = tp.model.x0;tp.model.Qo = tp.model.Q0;
params.xo = params.model.x0;params.model.Qo = params.model.Q0;
[tpseq, tpLL] = exactInferenceLDS(seqOrig, tp,'getLL',true);
[seq,   LL]   = exactInferenceLDS(seqOrig, params,'getLL',true);

sum(abs(tpLL-LL))./sum(abs(tpLL))

params.model.C'*params.model.C
dlyap(params.model.A,params.model.Q)
%}