function [wEB,hprsML,logEv,postHess] = findEBestimate_GLM(w0,hprs0,mstruct)
% [wEB,hprsML,logEv,postHess] = findEBestimate_GLM(w0,hprs0,mstruct)
%
% Find empirical Bayes estimate for GLM regression weights (Poisson or
% logistic). Maximizes marginal likelihood (computed using Laplace
% approximation) for hyperparameters and returns MAP estimate given those
% hyperparameters 
%
% INPUTS:
%      w0 [m x 1] - initial guess at parameters 
%   hprs0 [p x 1] - initial guess at hyper-parameters
% mstruct [1 x 1] - model structure with fields:
%         .neglogli - func handle for negative log-likelihood
%         .logprior - func handle for log-prior 
%         .liargs - cell array with args to neg log-likelihood
%         .priargs - cell array with args to log-prior function
%
% OUTPUTS:
%        w [m x 1] - empirical Bayes parameter estimate for regression weights
%     hprs [p x 1] - maximum marginal likelihood hyper-parameter estimate
%    logEv [1 x 1] - log-evidence at maximum
% postHess [1 x 1] - Hessian (2nd derivs) of negative log-posterior at
%                    maximum ( inverse of posterior covariance)
%
% Note: this implementation is not especially robust or efficient, and
% should only be used when initialized somewhere in the vicinity of the ML
% hyperparameter values (e.g., identified by a coarse grid search).
%
% Parametrization: log transforms first hyperparameter (assumed to be precision),
% and logit transforms remaining hyperprs (assumed to be in [0,1]).
%
% $Id$

% find MAP estimate for w given initial hyperparams
opts1 = struct('tolX',1e-10,'tolFun',1e-10,'maxIter',1e4,'verbose',0);
lfpost = @(w)(neglogpost_GLM(w,hprs0,mstruct)); % loss func handle
wmap0 = fminNewton(lfpost,w0,opts1); % do optimization for w_map

% transform hyperparameters to Reals
htprs0 = transformhprs(hprs0,1);

% Find maximum of evidence as a function of hyperparams
opts2 = struct('tolX',1e-10,'tolFun',1e-10,'maxIter',25,'verbose',0); % opts governing MAP search
hloss = @(ht)(updateMAPandEvalEvidence(ht,wmap0,opts2,mstruct)); % neg evidence func handle
opts_fminunc = optimset('display', 'iter', 'largescale', 'off','maxfunevals',1e3,...
    'FinDiffType','central'); % opts governing evidence search
[htprsML,neglogEv1] = fminunc(hloss,htprs0,opts_fminunc); % maximize evidence

% get MAP estimate at these hyperparams
[neglogEv2,wEB,postHess] = updateMAPandEvalEvidence(htprsML,wmap0,opts1,mstruct);
if abs(neglogEv1-neglogEv2)>.1
    warning('evidence estimate may be inaccurate (findEBestimate_bernoulliGLM)');
end

% un-transform hyperparameters to appropriate range
hprsML = transformhprs(htprsML,-1);
logEv = -neglogEv2;

    

% ======================================================================
% Sub-function to update MAP parameters and evaluate evidence
% ======================================================================

function [neglogEv,wmap,postHess] = updateMAPandEvalEvidence(htprs,wmap0,opts,mstruct)
% [neglogEv,wmap] = updateMAPandEvalEvidence(htprs,X,Y,fprior,wmap0,opts)

% un-transform hyperparameters
hprs = transformhprs(htprs,-1);

% find MAP estimate for w
lfpost = @(w)(neglogpost_GLM(w,hprs,mstruct));
wmap = fminNewton(lfpost,wmap0,opts);

% evaluate negative log evidence using Laplace approxmiation
neglogEv = -logevid_GLM(wmap,hprs,mstruct);

% Compute Hessian of negative log-posterior 
if nargout > 2
    [~,~,postHess] = lfpost(wmap);
end

    