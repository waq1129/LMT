function [p,dp,negCinv,logdetrm] = logprior_ridge(prvec,rho,iirdge,rhoNull)
% [p,dp,negCinv,logdetrm] = logprior_ridge(prvec,rho,iirdge,rhoNull)
%
% Evaluate a Gaussian log-prior at parameter vector prvec.
%
% Inputs:
%   prvec [n x 1] - parameter vector (last element can be DC)
%     rho [1 x 1] - ridge parameter (precision)
%  iirdge [v x 1] - indices to apply ridge prior to
% rhoNull [1 x 1] - prior precision for other elements
%
% Outputs:
%      p [1 x 1] - log-prior
%     dp [n x 1] - grad
%   negCinv [n x n] - inverse covariance matrix (Hessian)
% logdet [1 x 1] - log-determinant of -negCinv (optional)
%
% $Id$

nx = size(prvec,1);

if nargin < 3
    iirdge = (1:nx)';
    rhoNull = 1;
elseif nargin < 4
    rhoNull = 1;
end

Cinvdiag = ones(nx,1)*rhoNull;
Cinvdiag(iirdge) = rho;

dp = -bsxfun(@times,prvec,Cinvdiag);
p = .5*sum(bsxfun(@times,dp,prvec),1);

if nargout > 2
    negCinv = spdiags(-Cinvdiag,0,nx,nx);
    logdetrm = sum(log(Cinvdiag));
end
