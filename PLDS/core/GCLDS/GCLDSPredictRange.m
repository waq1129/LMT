function [ypred, xpred, xpredCov, seqInf] = GCLDSPredictRange(params,y,condRange,predRange,varargin)
% 
% function [ypred xpred xpredCov] = GCLDSPredictRange(params,y,condRange,predRange,varargin)
%
% arguments:
% y         yDim x T, spike data
% condRange time used to conditioning on (must be contained in y)
% predRange time to predict 
%
% Yuanjun Gao, Lars Buesing, 2015
%

s       = [];
u       = [];
lamInit = [];

assignopts(who,varargin);
 
[yDim xDim] = size(params.model.C);

Tcond = numel(condRange);
Tpred = numel(predRange);
tcRlo = min(condRange);
tcRhi = max(condRange);
tpRlo = min(predRange);
tpRhi = max(predRange);

if size(y,2)<tcRhi
   error('Conditioning range larger than data, aborting')
elseif size(y,2)>Tcond
   y = y(:,condRange);
   if params.model.notes.useB
     ucond = u(:,condRange);
   end
   if params.model.notes.useS
     scond = s(:,condRange);
   end
end


paramsInf = params;
if paramsInf.model.notes.useB; paramsInf.model.x0 = paramsInf.model.x0+paramsInf.model.B*u(:,1);end
for t=2:tcRlo                                                   % get the starting distribution right
  paramsInf.model.x0 = paramsInf.model.A*paramsInf.model.x0;
  if paramsInf.model.notes.useB&&(t<tcRlo); paramsInf.model.x0 = paramsInf.model.x0+paramsInf.model.B*u(:,t);end
  paramsInf.model.Q0 = paramsInf.model.A*paramsInf.model.Q0*paramsInf.model.A'+paramsInf.model.Q;
end

seqInf.y  = y;
seqInf.T  = size(seqInf.y,2);
if numel(lamInit)==(yDim*Tcond)
   disp('Warm-starting predction inference')
   seqInf.posterior.lamOpt = lamInit;
end
if paramsInf.model.notes.useB; seqInf.u = ucond; end;
if paramsInf.model.notes.useS; seqInf.s = scond; end;

seqInf   = params.model.inferenceHandle(paramsInf,seqInf);
xpred    = zeros(xDim,Tpred);
xpredCov = zeros(xDim,xDim,Tpred);
ypred    = zeros(yDim,Tpred);

xNow     = seqInf(1).posterior.xsm(:,end);
xCovNow  = seqInf(1).posterior.Vsm(end+1-xDim:end,:); 

g = params.model.g;
K = size(g, 2);
for t = (tcRhi+1):tpRhi    % progagate prediction
    xNow    = paramsInf.model.A*xNow;
    if params.model.notes.useB; xNow = xNow+params.model.B*u(:,t); end;
    xCovNow = paramsInf.model.A*xCovNow*paramsInf.model.A'+paramsInf.model.Q;
    if t>=tpRlo
       xpred(:,t-tpRlo+1) = xNow;
       xpredCov(:,:,t-tpRlo+1) = xCovNow;
       yr = params.model.C*xNow+0.5*diag(params.model.C*xCovNow*params.model.C');
       if params.model.notes.useS
	 yr = yr+s(:,t);
       end
       
       
       fact_seq = cumsum(log(1:K));
    g_fact = bsxfun(@minus, g, fact_seq);
    log_p_raw = bsxfun(@plus, bsxfun(@times, exp(yr), 1:K), g_fact);
    log_p_max = max(max(log_p_raw, [], 2), 0);
    log_p_raw = bsxfun(@minus,log_p_raw, log_p_max);
    p_raw = exp(log_p_raw);
    p_raw = [exp(-log_p_max), p_raw];
    p_normalizer = sum(p_raw, 2);
    p_norm = bsxfun(@times, p_raw, 1./p_normalizer);
    yHat = sum(bsxfun(@times, 0:K, p_norm),2);
       ypred(:,t-tpRlo+1) = yHat;
    end
end
