function [p,dp,negCinv,logdetrm] = logprior_AR1(prvec,theta,smoothingGroups,rhoNull)
% [p,dp,negCinv,logdetrm] = logprior_AR1(prvec,theta,nx,rhoNull)
%
% Evaluate Gaussian AR1 log-prior at parameter vector prvec.
%
% Inputs:
%  prvec [n x 1] - parameter vector (last element can be DC)
%  theta [2 x 1] - [rho; (precision)
%                   alpha (smoothness) ]
%     nx [1 x 1] - length of param vector to apply to prior (last element
%                  of prvec can be all-ones for dc term
%  rhoNull [1 x 1] - prior precision for DC term (optional)
%
% Outputs:
%         p [1 x 1] - log-prior
%        dp [n x 1] - grad
%   negCinv [n x n] - negative of inverse covariance matrix (Hessian)
% logdet [1 x 1] - log-determinant of -negCinv (optional)
%
% Inverse prior covariance matrix given by:
%  C^-1 = rho/(1-a^2) [ 1 -a
%                      -a 1+a^2 -a                       
%                        .   .   .
%                          -a 1+a^2 -a
%                                -a  1 ]
%
% $Id$

MINVAL = 1e-6;

if nargin <= 2
    smoothingGroups = length(prvec);
end
if nargin < 4
    rhoNull = .1;
end

rho = max(theta(1),MINVAL);
aa = min(theta(2),1-MINVAL);

const = rho/(1-aa.^2);

for kGrp = 1:numel(smoothingGroups)
    nx = smoothingGroups(kGrp);
    vdiag = [1;ones(nx-2,1)+aa^2;1]*const;
    voffdiag = -ones(nx,1)*aa*const;
    negCinv{kGrp} = -spdiags([voffdiag,vdiag,voffdiag],-1:1,nx,nx);
end

% Add prior indep prior variance on DC coeff
if sum(smoothingGroups) < size(prvec,1)
negCinv{end+1} = -rhoNull*diag(ones(1,size(prvec,1) - sum(smoothingGroups)));
end
negCinv = blkdiag(negCinv{:});

dp = negCinv*prvec;
p = .5*sum(prvec.*dp,1);

if nargout>3
    logdetrm = nx*log(rho)-(nx-1)*log(1-aa.^2);
end
