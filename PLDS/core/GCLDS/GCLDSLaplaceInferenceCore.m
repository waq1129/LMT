function smooth = GCLDSLaplaceInferenceCore(data, params)
% Compute laplace-approximated posterior means and covariances for GCLDS
% (c) Yuanjun Gao, Evan Archer, 2014
% 
runinfo = params.runinfo;
mps = params.initparams;

[yDim, T] = size(data.y);
xDim = size(data.xxInit, 1);
K = size(mps.g, 2);

% initialize latent states to 0
%xx = randn(runinfo.nStateDim, size(data.y,2)); % for subsequent EM iterations in the poisson case we'll want to initialize with previous x's.

xx = data.xxInit;

if params.notes.useB
  H = mps.B*data.u;% zeros(size(xx)); %mps.nlin.f(mps.nlin, data.h); % compute nlin first. 
else
  H = zeros(size(xx));
end

Qinv = pinv(mps.Q);
Q0inv = pinv(mps.initV);
AQiA = mps.A'*Qinv*mps.A;

LL = -inf;
LLprev = -inf;

%pre-computed variables for GPoisson likelihood
k_seq = 1:K;
fact = cumsum(log(k_seq)); %factorial sequence
g_fact = bsxfun(@minus, mps.g, fact);

while 1

XX = xx(:,2:end) - mps.A*xx(:,1:end-1) - H(:,2:end);
QiH = zeros(size(H));
QiH(:,2:end) = - Qinv*XX;
QiH(:,1) = - Q0inv*( xx(:,1) - H(:,1) - mps.initx );

T = size(data.y,2);

% sum over s and u
if params.notes.useS
    ds = data.s;
else
    ds = sparse(size(data.y,1), size(data.y,2));
end


%Ypred = bsxfun(@plus, mps.C*xx + ds, 0);%mps.d); %mps.d = 0 for now

%% Latent-state grad & hessian (this should be the same for both poisson and gaussian likelihoods)

lat_grad =  [ QiH(:,1:end-1) - mps.A'*QiH(:,2:end),   QiH(:,end) ];

II = speye(size(data.y,2)-2);
lat_hess_diag = -blkdiag(sparse(Q0inv+AQiA), kron( II, Qinv + AQiA), sparse(Qinv));

II_c = circshift(speye(size(data.y,2)), [0 1]); II_c(end,1) = 0; 
lat_hess_off_diag = kron(II_c, mps.A'*Qinv); lat_hess_off_diag = lat_hess_off_diag + lat_hess_off_diag';

lat_hess = lat_hess_diag  + lat_hess_off_diag;

%% GPoisson Observation gradient and hessian

theta = mps.C*xx + ds; %dim yDim x T

%calculate expectation of y

p_log_raw = bsxfun(@times, theta, reshape(k_seq,1,1,K)); %yDim x T x K
p_log_raw = bsxfun(@plus, p_log_raw, reshape(g_fact, yDim, 1, K));
p_max = max(max(p_log_raw,[], 3), 0);
p_log_raw = bsxfun(@minus, p_log_raw, p_max);
p_raw = exp(p_log_raw);
p_normalizer = sum(p_raw, 3) + exp(-p_max); %add back P(y=0)
p_norm = bsxfun(@rdivide, p_raw, p_normalizer);

epsilon = 1e-16;
while(any(vec(sum(p_norm, 3) > 1))),
    epsilon = epsilon * 10;
    p_norm = bsxfun(@rdivide, p_raw, p_normalizer + epsilon);
    warning('numerical unstable in calculating probability');
end

yHat = sum(bsxfun(@times, p_norm, reshape(k_seq,1,1,K)),3);
y2Hat = sum(bsxfun(@times, p_norm, reshape(k_seq.^2,1,1,K)),3);
yVar = y2Hat - yHat.^2;

YL = data.y-yHat;

YC = zeros(yDim, T, xDim);
zeros(runinfo.nObsDim, size(data.y,2), runinfo.nStateDim);
poiss_hess_blk = zeros(runinfo.nStateDim, runinfo.nStateDim, size(data.y,2));
for idx = 1:runinfo.nStateDim
   YC(:,:,idx) = bsxfun(@times, YL, mps.C(:,idx));
   poiss_hess_blk(idx,:,:) = -mps.C'*bsxfun(@times, yVar, mps.C(:,idx));
end

poiss_grad = sparse(reshape(sum(YC,1),T, xDim)');
poiss_hess = spblkdiag(poiss_hess_blk);

%% Add the latent and observation hessian & gradient

Hess = poiss_hess + lat_hess;

Grad = poiss_grad + lat_grad;


%% Compute newton step, perform line search

if LL - LLprev < 1e-10 % TODO: put tolerance in setup file
    break
end

xold = xx;
UPDATE  = reshape(Hess \ Grad(:), size(lat_grad));

dx = 0;
LLprev = LL;
ii = 0;

LOGDETS = logdet_plds(mps.initV) + (T-1)*logdet_plds(mps.Q);
while 1
    dx = 2^(-ii); ii = ii + .1;
    xx = xold - dx*UPDATE;
    
    %% Recompute just the likelihood @ dx step
    %Ypred = bsxfun(@plus, mps.C*xx + ds, mps.d);
    theta = mps.C*xx + ds;
    %Lambda = exp(Ypred);
    
    p_log_raw = bsxfun(@times, theta, reshape(k_seq,1,1,K)); %yDim x T x K
    p_log_raw = bsxfun(@plus, p_log_raw, reshape(g_fact, yDim, 1, K));
    p_max = max(max(p_log_raw, [], 3), 0);
    p_log_raw = bsxfun(@minus, p_log_raw, p_max);
    p_raw = exp(p_log_raw);
    p_normalizer = sum(p_raw, 3) + exp(-p_max); %add back P(y=0)
    
    if dx < .001
        break
    end

    % Let's see if we can't compute the likelihood. 
    % fprintf('\ncomputing likelihood:\n')
    
    XX = xx(:,2:end) - mps.A*xx(:,1:end-1) - H(:,2:end);
    X0 = (xx(:,1) - mps.initx - H(:,1));

    % Dropping constant terms that depend on dimensionality of states and observations
    % (might want to change that for model comparison purposes later) 
    PLL = sum(sum(data.y.*theta-p_max-log(p_normalizer)));
    GnLL = LOGDETS + sum(sum(XX.*(Qinv*XX))) + sum(sum(X0.*(Q0inv*X0)));

    LL = PLL - GnLL/2;
    %% Finish recomputing the likelihood
   
    if LL > LLprev
        break
    end
end

%fprintf('%0.2f --> %0.3f', dx, LL);

end

%%

AA0 = -(Q0inv + AQiA);
AAE = -Qinv;
AA = -(Qinv + AQiA);
BB = (mps.A'*Qinv); 
smooth = [];
smooth.VV = zeros(size(AA,1),size(AA,1),T);
AAz = poiss_hess_blk; AAz(:,:,1) = AA0 + AAz(:,:,1); AAz(:,:,2:end-1) = bsxfun(@plus, AA, poiss_hess_blk(:,:,2:end-1)); AAz(:,:,end) = AAE + AAz(:,:,end);

AAz = -AAz; BB = -BB;

[smooth.V,smooth.VV(:,:,2:end)]=sym_blk_tridiag_inv_v1(AAz, BB, [1:T]', ones(T-1,1));

smooth.x = xx;
smooth.loglik = LL;

smooth.lamOpt = reshape(reshape(p_norm, yDim*T, K)', yDim*T*K, 1);
%smooth.Ypred = Ypred;

%% With minfunc
% 
% opt = struct('Method','csd', 'maxFunEvals', 500, 'Display', 'on'); % TODO: Make this an option in opt struc
% xxmin = minFunc( @(x) poiss_obj_fun(x, data, params), xx(:), opt);


