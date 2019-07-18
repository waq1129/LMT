function [seq varBound Lambda LambdaPost] = LDSInference(params,seq)
%
% simplest Kalman smoother in O(T)
%
% assume all trials are of the same length T
%



[yDim xDim] = size(params.model.C);  

Trials      = numel(seq);
T           = size(seq(1).y,2);


%%%%%%%%%%%%% covariances %%%%%%%%%%%%

Lambda     = buildPriorPrecisionMatrixFromLDS(params,T); % prior precision
LambdaPost = Lambda;                                 % posterior precision

CRC    = zeros(xDim*T,xDim);
CRinvC = params.model.C'*pinv(params.model.R)*params.model.C;
for t=1:T
  xidx = ((t-1)*xDim+1):(t*xDim);
  CRC(xidx,:) = CRinvC;
  LambdaPost(xidx,xidx)  = LambdaPost(xidx,xidx) + CRinvC;
end

[Vsm, VVsm] = smoothedKalmanMatrices(params.model,CRC);


%%%%%%%%%%%%%%%%% means %%%%%%%%%%%%%%%%

varBound = -0.5*T*Trials*yDim;

Mu = getPriorMeanLDS(params,T);
LamMu = Lambda*vec(Mu);

for tr=1:Trials

  Y = bsxfun(@minus,seq(tr).y,params.model.d);
  if params.model.notes.useS
    Y = Y-seq.s;
  end
  Yraw = Y;
  Y  = params.model.R\Y;
  Y  = params.model.C'*Y;
  seq(tr).posterior.xsm  = reshape(LambdaPost\(LamMu+vec(Y)),xDim,T);
  seq(tr).posterior.Vsm  = Vsm;
  seq(tr).posterior.VVsm = VVsm;

  Yraw = Yraw-params.model.C*Mu;
  varBound = varBound - 0.5*trace(params.model.R\(Yraw*Yraw'));

  Yraw = vec(params.model.C'*(params.model.R\Yraw));
  varBound = varBound + 0.5*Yraw'*(LambdaPost\Yraw);

end

varBound = varBound - 0.5*Trials*(logdet_plds(LambdaPost,'chol')-logdet_plds(Lambda,'chol')+T*logdet_plds(params.model.R));