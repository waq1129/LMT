function [logev,kmap,LL] = gaussLogEvidence(CpriorInv,nsevar,XX,XY,YY,ny)
% COMPLOGEV - log-evidence under Gaussian likelihood and zero-mean prior
%
% [logev,kmap,LL] = gaussLogEvidence(CpriorInv,nsevar,XX,XY,YY,ny)
%  
% The Model: 
%       likelihood: P(Y|X,k) = N(Y; XK,nsevar)
%            prior: P(k) = N(0, Cprior);
%
% Inputs: 
%    CpriorInv MxM - inverse prior covariance matrix
%       nsevar 1x1 - likelihood variance
%           XX NxN - X'*X x
%           YY 1x1 - Y'*Y 
%           ny 1x1 - length(Y) 
%
% Outputs: 
%        logev 1x1 - log-evidence
%         kmap Mx1 - MAP estimate (posterior mean)
%           LL MxM - posterior covariance matrix
%
% Note: calling this function with 

% 1. Compute MAP estimate of filter
H = XX/nsevar + CpriorInv;
% LL = inv(H);  % Posterior Covariance
% kmap = LL*XY/nsevar;  % Posterior Mean (MAP estimate)
kmap = H\XY/nsevar;  % Posterior Mean (MAP estimate)

% 1st term, from sqrt of log-determinants
% trm1 = .5*(logdet(LL) + logdet(CpriorInv) - (ny)*log(2*pi*nsevar));
trm1 = .5*(-logdet(H) + logdet(CpriorInv) - (ny)*log(2*pi*nsevar));

% 2nd term, from exponent
trm2 = -.5*(YY/nsevar - (XY'*(H\XY))/nsevar.^2);
% trm2 = -.5*(YY/nsevar - (XY'*LL*XY)/nsevar.^2);
    
logev = trm1+trm2;

