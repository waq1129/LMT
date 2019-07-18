function [params, cost] = PLDSLinkFuncMStepObservation(params,seq)
%
% function params = PLDSLinkFuncMStepObservation(params,seq)
% 
% do MStep for observation model of PLDS with arbitrary link function 
% uses stochastic gradient descent 
%
% !!! to do: 
% - take out cost return in the end
%

expFlag  = false;% !!! for debugging
evalCost = false; % !!! for debugging!

Trials = numel(seq);
[yDim,xDim] = size(params.model.C);

MaxIter = params.opts.algorithmic.MStepObservation.minFuncOptions.MaxIter;
gamma0  = params.opts.algorithmic.MStepObservation.minFuncOptions.gamma0./sum([seq.T]);
gamma0  = gamma0./(1+gamma0*params.state.EMiter).^(1.5);
progTol = params.opts.algorithmic.MStepObservation.minFuncOptions.progTol;
etaCd   = params.opts.algorithmic.MStepObservation.minFuncOptions.etaCd;

cost = zeros(MaxIter,1); %!!!

% compute Cholesky decomp of posterior cov for sampling
for tr=1:Trials
  T = seq(tr).T;
  Vsm = reshape(seq(tr).posterior.Vsm',xDim,xDim,T);
  for t=1:T
    XChol{t,tr} = chol(Vsm(:,:,t));
  end 
end

% initialization
Cnow  = params.model.C;
dnow  = params.model.d;
CCnow = zeros(xDim.^2,yDim);


%%%%%%%%%%%%%%%%%% main SGD loop %%%%%%%%%%%%%%

for ii=1:MaxIter;for tr=1:Trials

  T = seq(tr).T;
  Y = seq(tr).y;
  Vsm = reshape(seq(tr).posterior.Vsm',xDim.^2,T);  
  
  % sample from the posterior
  Xsamp = seq(tr).posterior.xsm;
  for t=1:T; Xsamp(:,t) = Xsamp(:,t) + XChol{t,tr}'*randn(xDim,1);end

  % evaluate the gradient
  Z   = bsxfun(@plus,Cnow*Xsamp,dnow); if params.model.notes.useS; Z = Z + seq(tr).s; end;
  fZ  = params.model.linkFunc(Z);
  dfZ = params.model.dlinkFunc(Z);
  
  gradCd = (Y.*dfZ./fZ-dfZ)*[Xsamp;ones(1,T)]';
  if params.model.notes.useCMask; gradCd(:,1:end-1) = gradCd(:,1:end-1).*params.model.CMask; end  

  % update
  alpha = gamma0./(1+gamma0*ii);  % adapt learning rate
  Cnow = Cnow + alpha*gradCd(:,1:end-1);
  dnow = dnow + alpha*gradCd(:,end);
  
  % check termination condition
  if norm(gradCd,'fro')<progTol
    progReached = true;
    break;
  else
    progReached = false;
  end

  if evalCost
    % evaluate true cost function; don't do this in the final method
    for yd=1:yDim; CCnow(:,yd) = vec(Cnow(yd,:)'*Cnow(yd,:));end
    ZDet = bsxfun(@plus,Cnow*seq(tr).posterior.xsm,dnow);
    if params.model.notes.useS; ZDet = ZDet+seq(tr).s; end
    ZVar = CCnow'*Vsm;
    if expFlag
      cost(ii) = cost(ii) + sum(vec(Y.*ZDet-exp(ZDet+0.5*ZVar)));
    else
      cost(ii) = cost(ii) + sum(vec(computeEqlogPoissonFromTable(params,Y,ZDet,sqrt(ZVar))));
    end
  end

end;if progReached; break; end;end

if params.model.notes.useCMask; Cnow = Cnow.*params.model.CMask; end
params.model.C = etaCd*params.model.C+(1-etaCd)*Cnow;
params.model.d = etaCd*params.model.d+(1-etaCd)*dnow;

