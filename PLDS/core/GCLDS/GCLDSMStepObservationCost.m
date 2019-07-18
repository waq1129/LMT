function [f, df] = GCLDSMStepObservationCost(vecCg,seq,params,lam_concave)
%
% function [f, df] = PLDSMStepObservationCost(vecCd,seq,params)
%
% Mstep for observation parameters C,g for standard GCLDS with exp-GPoisson observations
%
% Input:
%	- a vector containing information for C and g, length and format vary
%	depending on different assumptions for g
%
% (c) Y Gao, L Buesing 2014


Trials  = numel(seq);
yDim    = size(seq(1).y,1);
xDim    = size(params.model.A,1);
K = size(params.model.g, 2);
if ~exist('lam_concave', 'var') || isempty(lam_concave), lam_concave = 0; end

if params.model.notes.gStatus == 2, %share same curvature but different slope
    C = reshape(vecCg(1:(yDim * xDim)), yDim, xDim);
    d = reshape(vecCg((yDim*xDim+1):(yDim*(xDim+1))), yDim, 1);
    g_adjusted = reshape(vecCg((yDim*(xDim+1)+1):end), 1, []);
    g = bsxfun(@plus, bsxfun(@times, d, 1:K), [0,g_adjusted]);
elseif params.model.notes.gStatus == 1, %share everything
    C = reshape(vecCg(1:(yDim * xDim)), yDim, xDim);
    g = repmat(reshape(vecCg((yDim*xDim+1):end), 1, K), yDim, 1);
elseif params.model.notes.gStatus == 0,
    CgMat   = reshape(vecCg,yDim,xDim+K); %different curvature
    C       = CgMat(:,1:xDim);
    g       = CgMat(:,(xDim+1):end);
    %g_Mask  = hist([seq.y]', 1:K)' == 0;
    %g(g_Mask) = -Inf;
elseif params.model.notes.gStatus == 3,
    CgMat   = reshape(vecCg,yDim,xDim+1); %just a truncated PLDS, truncated at the same place
    C       = CgMat(:,1:xDim);
    g       = bsxfun(@times, CgMat(:,end), 1:K);  
end

g_ridge_lam = max(0, params.model.notes.g_ridge_lam);

if lam_concave > 0 && K >= 2, %check concavity
    g_full = [zeros(yDim,1), g];
    g_diff = diff(g_full, 2, 2);
    g_diff(isnan(g_diff)) = -Inf;
    if nanmax(g_diff(:)) >= 0,
        f = inf; df = nan(size(vecCg));
        disp('result not concave');
        return;
    end
end
    

if params.model.notes.useCMask; C = C .* params.model.CMask; end

CC      = zeros(yDim,xDim^2);
for yd=1:yDim
  CC(yd,:) = vec(C(yd,:)'*C(yd,:));
end


f   = 0;				% current value of the cost function = marginal likelihood
df  = zeros(size(C));			% derviative wrt C
dfg = zeros(yDim,K);			% derivative wrt g

for tr=1:Trials
 
  T    = size(seq(tr).y,2);
  y    = seq(tr).y;
  y_count = hist(y', 0:K)';
  if(yDim == 1)
      y_count = y_count';
  end
  y_count = y_count(:,2:end);
  m    = seq(tr).posterior.xsm;
  Vsm  = reshape(seq(tr).posterior.Vsm',xDim.^2,T);
  
  d = repmat(vec(g'), T, 1) - repmat(cumsum(log(1:K)'), T*yDim, 1);
  h    = C*m;
  if params.model.notes.useS; h = h+seq(tr).s; end
  rho  = CC*Vsm;

  k_seq = reshape(1:K,1,1,K); %make a sequence of dimension three
  
  %not really log_p but something similar in variational inference
  log_p_raw = bsxfun(@times, h, k_seq) + bsxfun(@times, rho, k_seq.^2 / 2);
  log_p_raw = bsxfun(@plus, log_p_raw, reshape(g, yDim, 1, K));
  log_p_raw = bsxfun(@minus, log_p_raw, reshape(cumsum(log(1:K)), 1, 1, K));
  log_p_max = max(max(log_p_raw, [], 3),0);
  p_raw = exp(bsxfun(@minus, log_p_raw, log_p_max));
  normalizer = sum(p_raw, 3) + exp(-log_p_max); %last term is log(1) - exp(log_p_max)
  p_norm = bsxfun(@times, p_raw, 1./normalizer);
  yhat = sum(bsxfun(@times, p_norm, k_seq), 3);
  y2hat = sum(bsxfun(@times, p_norm, k_seq.^2), 3);

  %yhat = exp(h+rho/2);
  f    = f + sum(vec(y.*h-log(normalizer)-log_p_max)) + nansum(vec(y_count .* g));
  
  TT   = y2hat*Vsm';
  TT   = reshape(TT,yDim*xDim,xDim);
  TT   = reshape((sum(reshape(bsxfun(@times,TT,vec(C)),yDim,xDim,xDim),2)),[],xDim);
     
  df   = df  + (y-yhat)*m'-TT;
  dfg  = dfg + y_count - reshape(sum(p_norm,2), yDim, K);
  
end



if lam_concave > 0 && K >= 2,
    %g_diff(isinf(g_diff)) = -Inf;
    g_diff_inv = 1./g_diff;
    g_diff(isinf(g_diff)) = nan;
    f = f + nansum(log(-g_diff(:))) * lam_concave;
    dg_conv = conv2(-g_diff_inv, [-1,2,-1]);
    dg_conv(:,1) = [];
    dfg = dfg + dg_conv * lam_concave;
end

f  = -f;
if params.model.notes.useCMask; df = df.*params.model.CMask; end

% add penalty for 2nd difference of g functions
if g_ridge_lam > 0, %params.model.notes.gStatus == 6, %add ridge penalty
    g_full = [zeros(yDim,1), g];
    g_diff = diff(g_full, 2, 2);
    f = f + g_ridge_lam * sum(g_diff(:).^2);
end

% merge the gradient
df = -df;
dfg = -dfg;

if K >= 2 && g_ridge_lam > 0,
    g_diff_full = [zeros(yDim, 1), g_diff, zeros(yDim, 2)];
    dfg = dfg + 2 * g_ridge_lam * g_diff_full(:,1:K);
    dfg = dfg - 4 * g_ridge_lam * g_diff_full(:,2:(K+1));
    dfg = dfg + 2 * g_ridge_lam * g_diff_full(:,3:(K+2));
end

if params.model.notes.gStatus == 2,
    dfd = sum(bsxfun(@times, dfg, 1:K), 2);
    dfg = sum(dfg(:,2:end),1);
    df = [vec(df); vec(dfd); vec(dfg)];
elseif params.model.notes.gStatus == 1,
    dfg = sum(dfg);
    df = [vec(df); vec(dfg)];
elseif params.model.notes.gStatus == 0,
    df = vec([df dfg]);
elseif params.model.notes.gStatus == 3,
    dfd = sum(bsxfun(@times, dfg, 1:K), 2);
    df = [vec(df); vec(dfd)];
end
