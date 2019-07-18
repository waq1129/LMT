function params = LDSMStepLDS(params,seq)
%
% function params = MStepLDS(params,seq)
%
% Parameters to update: A,Q,Q0,x0,B
%
%
% (c) L Buesing 2014
%

xDim    = size(params.model.A,1);
Trials  = numel(seq);


%% compute posterior statistics

if params.model.notes.useB
  uDim = size(seq(1).u,1);
else
  uDim = 0;
end

S11 = zeros(xDim,xDim);
S01 = zeros(xDim+uDim,xDim);
S00 = zeros(xDim+uDim,xDim+uDim);

x0 = zeros(xDim,Trials);
Q0 = zeros(xDim,xDim);

Tall = [];

for tr = 1:Trials

    T = size(seq(tr).y,2);
    Tall  = [Tall T];

    if isfield(seq(tr).posterior,'Vsm')
      Vsm   = reshape(seq(tr).posterior.Vsm' ,xDim,xDim,T);
      VVsm  = reshape(seq(tr).posterior.VVsm',xDim,xDim,T-1);
    else
      Vsm   = reshape(seq(1).posterior.Vsm' ,xDim,xDim,T);
      VVsm  = reshape(seq(1).posterior.VVsm',xDim,xDim,T-1);
    end

    MUsm0 = seq(tr).posterior.xsm(:,1:T-1);
    MUsm1 = seq(tr).posterior.xsm(:,2:T);

    S11                = S11                + sum(Vsm(:,:,2:T),3)  + MUsm1*MUsm1';
    S01(1:xDim,:)      = S01(1:xDim,:)      + sum(VVsm(:,:,1:T-1),3) + MUsm0*MUsm1';
    S00(1:xDim,1:xDim) = S00(1:xDim,1:xDim) + sum(Vsm(:,:,1:T-1),3)  + MUsm0*MUsm0';

    if params.model.notes.useB
      u = seq(tr).u(:,1:T-1);
      S01(1+xDim:end,:)          = S01(1+xDim:end,:)          + u*MUsm1';
      S00(1+xDim:end,1:xDim)     = S00(1+xDim:end,1:xDim)     + u*MUsm0';
      S00(1:xDim,1+xDim:end)     = S00(1:xDim,1+xDim:end)     + MUsm0*u';
      S00(1+xDim:end,1+xDim:end) = S00(1+xDim:end,1+xDim:end) + u*u';
    end

    x0(:,tr) = MUsm0(:,1);
    Q0 = Q0 + Vsm(:,:,1);

end

S00 = (S00+S00')/2;
S11 = (S11+S11')/2;

if params.model.notes.learnA
  params.model.A  = S01'/S00;
end
params.model.Q  = (S11+params.model.A*S00*params.model.A'-S01'*params.model.A'-params.model.A*S01)./(sum(Tall)-Trials);
%params.model.Q  = S11-S01'/S00*S01;
params.model.Q  = (params.model.Q+params.model.Q')/2;

%[aQ bQ] = eig(params.model.Q);
%params.model.Q = aQ*diag(max(diag(bQ),0))*aQ';

if params.model.notes.useB
  params.model.B = params.model.A(:,1+xDim:end);
  params.model.A = params.model.A(:,1:xDim);
end

if params.model.notes.learnx0
  params.model.x0 = mean(x0,2);
end

x0dev = bsxfun(@minus,x0,params.model.x0);

if params.model.notes.learnQ0
  params.model.Q0 = (Q0 + x0dev*x0dev')./Trials;
else
  params.model.Q0 = dlyap(params.model.A,params.model.Q);
end

if (min(eig(params.model.Q))<0) || (min(eig(params.model.Q0))<0)
   keyboard
   params.model.Q=params.model.Q+1e-9*eye(xDim);
end