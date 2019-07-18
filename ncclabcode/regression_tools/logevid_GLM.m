function logEv = logevid_GLM(wts,hprs,mstruct)
% logEv = logevid_GLM(wts,hprs,mstruct)
%
% Compute negative log-evidence under a GLM using the Laplace approximation
%
% INPUTS:
%     wts [m x 1] - regression weights
%    hprs [p x 1] - hyper-parameters
% mstruct [1 x 1] - model structure with fields:
%         .neglogli - func handle for negative log-likelihood
%         .logprior - func handle for log-prior 
%         .liargs - cell array with args to neg log-likelihood
%         .priargs - cell array with args to log-prior function
%
% OUTPUT:
%   logEv - log-evidence using the Laplace approximation at wts (assumed to
%           be the MAP estimate given current hyperparameters)
%
% $Id$

[L,~,ddL] = mstruct.neglogli(wts,mstruct.liargs{:});
[p,~,negCinv,logdetCinv] = mstruct.logprior(wts,hprs,mstruct.priargs{:});

logEv = -L + p + .5*logdetCinv - .5*logdet(ddL-negCinv);

    