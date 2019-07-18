function seq = LDSsample(params,T,Trials,varargin)
%
% sampleLDS(params,T,Trials)
%
% sample from linear dynamical system model
%
%
% (c) L Buesing 2014 
%

u = [];
s = [];
assignopts(who,varargin);

if numel(T)==1
   T = ones(Trials,1)*T;
end

Trials = numel(T);

[yDim xDim] = size(params.model.C);
CQ          = chol(params.model.Q);
CQ0         = chol(params.model.Q0);

if isfield(params.model,'R') && params.model.notes.useR
  R = params.model.R;
  if size(R,2)==1
    R = diag(R);
  end
  CR = chol(R);
else
  CR = zeros(yDim);
end


for tr=1:Trials

  if params.model.notes.useB
    % !!! take this out
    if isempty(u)
%      warning('You are using an extremely dirty convenience function. It will disappear')
      uDim = size(params.model.B,2);
      gpsamp = sampleGPPrior(1,T(tr),uDim-1,'tau',10);
      tpsamp = (vec(repmat(rand(1,floor(T(tr)/10))>0.5,10,1))-0.5); tpsamp = [tpsamp' zeros(1,T(tr)-floor(T(tr)/10)*10)];
      seq(tr).u = [gpsamp{1}/3;tpsamp];
    else
      seq(tr).u = u{tr};
    end
  end

  seq(tr).x = zeros(xDim,T(tr));
  seq(tr).x(:,1) = params.model.x0+CQ0'*randn(xDim,1);
  if params.model.notes.useB; seq(tr).x(:,1) = seq(tr).x(:,1)+params.model.B*seq(tr).u(:,1);end;
  for t=2:T(tr)
      seq(tr).x(:,t) = params.model.A*seq(tr).x(:,t-1)+CQ'*randn(xDim,1);
      if params.model.notes.useB; seq(tr).x(:,t) = seq(tr).x(:,t)+params.model.B*seq(tr).u(:,t);end;
  end
  seq(tr).y = bsxfun(@plus,params.model.C*seq(tr).x,params.model.d)+CR'*randn(yDim,T(tr));
  seq(tr).T = T(tr);

  if params.model.notes.useS
    seq(tr).y = seq(tr).y+s{tr};
    seq(tr).s = s{tr};
  end

end



