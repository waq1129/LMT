function params = PLDSMStepNeuronPSTH(params,seq)
%
% params = PLDSMStepNeuronRFs(params,seq)
%

Trials = numel(seq);
[yDim T] = size(seq(1).y);
xDim = size(params.model.C,2);

Y = [seq.y];
Y = reshape(Y,yDim,T,Trials);

over_m = []; for tr=1:Trials; over_m = [over_m params.model.C*seq(tr).posterior.xsm];end
over_m = bsxfun(@plus,over_m,params.model.d);
over_m = reshape(over_m,yDim,T,Trials);

over_v = [];
for tr=1:Trials;
  T = size(seq(1).y,2);
  for t=1:T
    over_v = [over_v diag(params.model.C*seq(tr).posterior.Vsm((t-1)*xDim+1:t*xDim,:)*params.model.C')];
  end
end
over_v = reshape(over_v,yDim,T,Trials);

Sinit = params.model.S;
lam   = params.opts.algorithmic.MStepNeuronPSTH.lam;
opts  = params.opts.algorithmic.MStepNeuronPSTH.minFuncOptions;

Sopt = zeros(size(Sinit));
for yd=1:yDim
  ydat = squeeze(Y(yd,:,:));
  mdat = squeeze(over_m(yd,:,:));
  vdat = squeeze(over_v(yd,:,:));  
  Sopt(yd,:) = minFunc(@PLDSMStepNeuronPSTHCost,Sinit(yd,:)',opts,ydat,mdat,vdat,lam);;
end

% recompute seq.s

params.model.S = Sopt;
for tr=1:Trials
  seq(tr).s = Sopt;
end




