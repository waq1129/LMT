function [seq varBound] = PLDSLinkFuncLaplaceInference(params,seq)
%
% function [seq varBound] = PLDSLinkFuncLaplaceInference(params,seq)
%
% simplest Kalman smoother in O(T)
%
% performance tip:
% sort trials such that those with same T are next to each other,
% this ensures that computations are recycled
%

progTol         = params.opts.algorithmic.VarInfX.minFuncOptions.progTol;
MaxIter         = params.opts.algorithmic.VarInfX.minFuncOptions.MaxIter;
computeVarBound = true;
varBound        = 0;

linkFunc    = params.model.linkFunc;
dlinkFunc   = params.model.dlinkFunc;
d2linkFunc  = params.model.d2linkFunc;

[yDim xDim] = size(params.model.C);  
Trials      = numel(seq);
CC          = zeros(xDim.^2,yDim);
for yd=1:yDim; CC(:,yd) = vec(params.model.C(yd,:)'*params.model.C(yd,:));end

Tlast = 0;
for tr=1:Trials
  T = size(seq(tr).y,2);

  % allocate arrays
  if T~=Tlast; Lambda = buildPriorPrecisionMatrixFromLDS(params,T);end; % prior precision
  
  Y         = [seq(tr).y];
  Mu        = getPriorMeanLDS(params,T,'seq',seq(tr)); 
  LambdaMu  = Lambda*vec(Mu);
  progX     = inf;
  if isfield(seq(tr),'posterior') && isfield(seq(tr).posterior,'xsm')
    X = vec(seq(tr).posterior.xsm);
  else
    X = vec(Mu);
  end
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% inner loop of Laplace inference
  niter = 0;
  while ( (progX>progTol) && (niter<MaxIter) )  

    niter = niter+1;
    Hessian = Lambda;

    Z = bsxfun(@plus,params.model.C*reshape(X,xDim,T),params.model.d);
    if params.model.notes.useS; Z = Z+seq(tr).s; end
    
    dYhat   = dlinkFunc(Z);
    Yhat    = linkFunc(Z)+1e-10;
    YYhat   = (Y./Yhat);
    YYhatdf = YYhat.*dYhat;
    d2linkFuncZ = d2linkFunc(Z);
    d2linkFuncZ(isnan(d2linkFuncZ)) = 0;
    
    % gradient
    gradX  = LambdaMu - Lambda*X + vec(params.model.C'*(YYhatdf-dYhat));

    % Hessian
    CRinvCloc = reshape(CC*(-d2linkFuncZ.*(YYhat-1)+YYhatdf.*dYhat./Yhat),xDim,xDim,T);
    xidx = 1:xDim;
    for t=1:T
      Hessian(xidx,xidx) = Hessian(xidx,xidx) + CRinvCloc(:,:,t);
      xidx = xidx+xDim;
    end

    Hessian = Hessian+0.001*eye(xDim*T);

    % Update
    if any(isnan(vec(Hessian)));
      keyboard
    end
    dX = Hessian\gradX;
    progX = norm(dX);
    X = X + dX;
    
    %{
    % eval cost for debugging
    Xmat = reshape(X,xDim,T);
    Zmat = bsxfun(@plus,params.model.C*Xmat,params.model.d); 
    fval = [fval -0.5*vec(Xmat-Mu)'*Lambda*vec(Xmat-Mu)+sum(vec([seq(tr).y].*log(linkFunc(Zmat))-linkFunc(Zmat)))];
    %fprintf('fval = %d, ||dX|| = %d\n',fval(end),progX);
    %}

  end

  seq(tr).posterior.xsm = reshape(X,xDim,T);
  [seq(tr).posterior.Vsm, seq(tr).posterior.VVsm] = smoothedKalmanMatrices(params.model,reshape(permute(CRinvCloc,[1 3 2]),xDim*T,xDim));
  
  Tlast = T;
  
  if computeVarBound 

    % determine D_KL[q||p]
    seq(tr).varBound = 0.5*logdet_plds(Lambda,'chol')-0.5*vec(seq(tr).posterior.xsm-Mu)'*Lambda*vec(seq(tr).posterior.xsm-Mu);
    seq(tr).varBound = seq(tr).varBound-0.5*logdet_plds(Hessian,'chol')-0.5*trace(Lambda/Hessian)+T*xDim/2; 

    % likelihood contribution
    ZStd = sqrt(CC'*reshape(seq(tr).posterior.Vsm',xDim.^2,T));
    explogPoissonLike = computeEqlogPoissonFromTable(params,Y,Z,ZStd);

    seq(tr).posterior.varBound = seq(tr).varBound+sum(vec(explogPoissonLike));
    varBound = varBound+seq(tr).posterior.varBound;

  end
  
end


