function [kRidge,alpha,nsevar] = autoRidgeRegress(xx,xy,yy,ny,opts,alpha0,nsevar0)
% [kRidge,alpha,nsevar] = autoRidgeRegress(xx,xy,yy,ny,opts,alpha0,nsevar0)
%
% Computes empirical Bayes ridge regression filter estimate.
%
% Estimates prior variance by maximizing evidence and returns the MAP
% filter estimate given this prior.
%
% Inputs:
%   xx - unnormalized autocorrelation of design matrix
%   xy - unnormalized crosscorrelation between design matrix and dependent var
%   yy - squared sum of dependent variable
%   ny - number of samples that computed yy
%   opts (optional) = options stucture: fields 'maxiter', 'tol', 'maxalpha'
%   alpha0 (optional) = initial value for precision hyperparameter
%   nsevar0 (optional) = initial value for noise variance 
%
% Outputs:
%   kRidge = empirical bayes estimate of kernel (under "ridge" prior).
%   alpha - estimate for prior precision (inverse variance)
%   nsevar - estimate for noise variance (per effective bins in ny)
%
% Note: traditional ridge parameter lambda = alpha*nsevar.

% Check that options field is passed in
if nargin <= 5  
    opts.maxiter = 1000;    % max # of ARD iterations
    opts.tol = 1e-9;      % stop if param change is below this
    opts.maxalpha = 1e8;  % alpha above which we'll regard the filter as all-zeros
    opts.verbose  = true;
    %fprintf('autoRidgeRegress: setting EB ridge regression opts to defaults\n');
end

% --- initialize some stuff -------
nx = size(xx,1); % length of stimulus filter
Lmat = speye(nx);  % diagonal matrix for prior
jcount = 1;      % counter for fixed-point algorithm
dparams = inf;   % change in params from previous step (initialize to inf)

% --- initialize k, alpha and noisevar, if necessary
if nargin <=6
    lam0 = 1;        % initial value of ridge parameter
    kmap0 = (xx + lam0*Lmat)\xy;
    nsevar = (yy - 2*kmap0'*xy + kmap0'*xx*kmap0)/ny;
    assert(nsevar > 0, 'Noise variance must be greater than 0');
    alpha = lam0/nsevar;
else
    alpha = alpha0;  % use initial values passed in
    nsevar = nsevar0;
end

% --- run fixed-point algorithm for maximizing marginal likelihood (1-param ARD)  ------
while (jcount <= opts.maxiter) && (dparams>opts.tol) && (alpha < opts.maxalpha)
    
    CpriorInv = Lmat*alpha; % form prior covariance matrix
    [mu,Ltrace] = compGaussianPosterior(xx,xy,nsevar,CpriorInv); % compute posterior mean, cov
    alpha2 = (nx - alpha.*Ltrace)./sum(mu.^2);  % new apha value
    resids = (yy  -  2*mu'*xy  +  mu'*xx*mu); % residual error
    nsevar2 = resids./(ny-nx+alpha*Ltrace);
    
    % update counter, alpha & nsevar
    dparams = norm([alpha2;nsevar2]-[alpha;nsevar]);
    jcount = jcount+1;
    alpha = alpha2;
    nsevar = nsevar2;

end

% check if we converged or not
if opts.verbose 
    if alpha >= opts.maxalpha
	fprintf(1, 'autoRidgeRegress: precision alpha=inf; filter is all-zeros (#%d steps)\n', jcount);
    elseif jcount < opts.maxiter
	fprintf(1, 'autoRidgeRegress: finished EB ridge regression in #%d steps\n', jcount)
    else
	fprintf(1, 'autoRidgeRegress: MAXITER (%d) steps; dparams=%f\n', jcount, dparams);
    end
end

kRidge = compGaussianPosterior(xx,xy,nsevar,Lmat*alpha); % compute MAP estimate for k
lambda = alpha*nsevar;  % ML estimate of the ridge parameter

% ============================================================
function [mu,Ltrace] = compGaussianPosterior(XX,XY,nsevar,CpriorInv)
% [mu,Ltrace,Lfull] = compGaussianPosterior(XX,XY,nsevar,CpriorInv)
%
% Compute posterior mean and variance given observed data Y given X, 
% (Or, given xx = X'*X; XY = X'*Y) and with
% noise variance nsevar and prior inverse-covariance CpriorInv

switch nargout
    case 1,
        mu = (XX+nsevar*CpriorInv)\XY;
    case 2,
        Linv = XX./nsevar + CpriorInv; % inverse covariance
        mu = Linv\(XY/nsevar);   % mean
        Ltrace = traceinv(Linv); % trace of covariance
end
