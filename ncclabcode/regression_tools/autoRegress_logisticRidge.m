function [wRidge,rho,SDebars,postHess,logevid] = autoRegress_logisticRidge(xx,yy,iirdge,rhoNull,rhovals,w0)
% [wRidge,rho,SDebars,postHess,logevid] = autoRegress_logisticRidge(xx,yy,iirdge,rhoNull,rhovals,w0)
%
% Computes empirical Bayes logistic ridge regression filter estimate under a
% Bernoulli-GLM with ridge prior
%
% Inputs:
%        xx [n x p] - stimulus (regressors)
%        yy [n x 1] - spike counts (response)
%     iirdg [q x 1] - weight indices for which to learn ridge parameter (columns of xx)
%   rhoNull [1 x 1] - fixed prior precision for other columns of xx
%   rhovals [v x 1] - list of prior precisions to use for grid search
%        w0 [p x 1] - initial estimate of regression weights (optional)
%   opts (optional) = options stucture: fields 'tolX', 'tolFun', 'maxIter','verbose'
%
% Outputs:
%   wRidge [p x 1] - empirical bayes estimate of weight vector
%      rho [1 x 1] - estimate for prior precision (inverse variance)
%  SDebars [p x 1] - 1 SD error bars from posterior
% postHess [p x p] - Hessian (2nd derivs) of negative log-posterior at
%                    maximum ( inverse of posterior covariance)
%  logevid [1 x 1] - log-evidence for hyperparameters
%
% $Id$

nw = size(xx,2); % length of stimulus filter

% initialize filter estimate with MAP regression estimate, if necessary
if nargin == 5
    rho0 = 5;        % initial value of ridge parameter
    Lprior = speye(nw)*rho0;
    w0 = (xx'*xx+Lprior)\(xx'*yy);
end

% --- set prior and log-likelihood function pointers ---
mstruct.neglogli = @neglogli_bernoulliGLM;
mstruct.logprior = @logprior_ridge;
mstruct.liargs = {xx,yy};
mstruct.priargs = {iirdge,rhoNull};

% --- Do grid search over ridge parameter -----
[hprsMax,wmapMax] = gridsearch_GLMevidence(w0,mstruct,rhovals);
fprintf('best grid point: rho (precision)=%.1f\n', hprsMax);

% --- Do gradient ascent on evidence ----
[wRidge,rho,logevid,postHess] = findEBestimate_GLM(wmapMax,hprsMax,mstruct);

if nargout >2
    SDebars = sqrt(diag(inv(postHess)));
end
