function [wRidge,hprs,SDebars,postHess,logevid] = autoRegress_logisticAR1_2D(xx,yy,fltdms,rhoNull,rhovals,avals,w0)
% [wRidge,hprs,SDebars,postHess,logevid] = autoRegress_logisticAR1_2D(xx,yy,fltdms,rhoNull,rhovals,avals,w0) 
%
% Computes empirical Bayes logistic ridge regression filter estimate under a
% Bernoulli-GLM with ridge prior
%
% Inputs:
%        xx [n x p] - stimulus (regressors)
%        yy [n x 1] - spike counts (response)
%    fltdms [1 x 2] - [ny, nx] size of 2D filter 
%   rhoNull [1 x 1] - fixed prior precision for other columns of xx
%   rhovals [v x 1] - vector of prior precisions to use for grid search
%     avals [v x 1] - vector of prior AR1 params to use for grid search
%        w0 [p x 1] - initial estimate of regression weights (optional)
%   opts (optional) = options stucture: fields 'tolX', 'tolFun', 'maxIter','verbose'
%
% Outputs:
%   wRidge [p x 1] - empirical bayes estimate of weight vector
%     hprs [2 x 1] - maximum-evidence hyperparameters (precision and AR1 param)
%  SDebars [p x 1] - 1 SD error bars from posterior
% postHess [p x p] - Hessian of neg log-posterior at EB estimate
%  logevid [1 x 1] - log-evidence of hyperparameters
%
% $Id$

nw = size(xx,2); % length of filter

% initialize filter estimate with MAP regression estimate, if necessary
if nargin == 5
    rh0 = 5;        % initial value of ridge parameter
    Lprior = speye(nw)*rhoNull;
    Lprior(iirdge,iirdge) = rho0;
    w0 = (xx'*xx+Lprior)\(xx'*yy);
end

% --- set prior and log-likelihood function pointers ---
mstruct.neglogli = @neglogli_bernoulliGLM;
mstruct.logprior = @logprior_AR1_2D;
mstruct.liargs = {xx,yy};
mstruct.priargs = {fltdms,rhoNull};

% --- Do grid search over ridge parameter -----
[hprsMax,wmapMax] = gridsearch_GLMevidence(w0,mstruct,rhovals,avals);
fprintf('best grid point: rho=%.0f, alpha=%.2f\n', hprsMax);

% --- Do gradient ascent on evidence ----
[wRidge,hprs,logevid,postHess] = findEBestimate_GLM(wmapMax,hprsMax,mstruct);

if nargout >2
    SDebars = sqrt(diag(inv(postHess)));
end
