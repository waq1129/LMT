function model = LDSObservedEstimation(X,model,dt,u)
%
% estimate paramaters of a fully observed LDS
%
% Input:
%
%   - X: observations   xDim x T
%   - model: struct to save parameters to
%   - dt: sub-sampling factor 
%   - u: external input (optional, only used if model.notes.useB == true)
%
% Ouput:
%
%   - model: standard LDS model structure 
%
%
% Lars Buesing, 2014

xDim = size(X,1);
T    = size(X,2);
Pi   = cov(X');
X    = bsxfun(@minus, X, mean(X, 2));

if ~model.notes.useB

  A  = X(:,2:end)/X(:,1:end-1);         % find A via regression
  if dt>1                               
    %A = diag(min(max((diag(abs(A))).^(1/dt),0),1)); 
    % clearly, this needs fixing !!!
     A = diag(min(max(diag(abs(A)),0.1),1));
  end
  Q  = Pi-A*Pi*A';                      % residual covariance

else

  uDim = size(u,1);
  if size(u,2)>size(X,2)
    uSub = subsampleSignal(u,dt);
  else 
    uSub = u;
  end
  AB = X(:,2:end)/[X(:,1:end-1);uSub(:,2:end)];
  A  = AB(1:xDim,1:xDim);
  B  = AB(1:xDim,1+xDim:end);

  if dt>1
    A = diag(min(max((diag(abs(A))).^(1/dt),0),1));

    uauto = zeros(uDim,dt);
    for ud=1:uDim
      uauto(ud,:) = autocorr(u(ud,:),dt-1);      
    end
    Aauto = zeros(xDim,dt);
    for tt=1:dt
      Aauto(:,tt) = diag(A).^(tt-1);
    end
    M = Aauto*uauto'*diag(1./uauto(:,1));
    B = B./M;
  end

  xuCov = A*X(:,1:end-1)*uSub(:,2:end)'/(size(uSub,2)-1)*B';
  Q = Pi-A*Pi*A'-B*cov(uSub')*B'-xuCov-xuCov';
  model.B = B;

end

[Uq Sq Vq] = svd(Q);                   % ensure that Q is pos def
Q  = Uq*diag(max(diag(Sq),0))*Uq';
x0 = zeros(xDim,1);
Q0 = dlyap(A,Q);


model.A  = A;
model.Pi = Pi;
model.Q  = Q;
model.Q0 = Q0;
model.x0 = x0;
