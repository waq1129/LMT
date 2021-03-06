function [p,dp,negCinv,logdetrm] = logprior_AR1fix(prvec,precision,aa,nx,rhoDC)
% [p,dp,negCinv,logdetrm] = logprior_AR1fix(prvec,precision,alpha)
%
% Evaluate Gaussian AR1 log-prior, with fixed AR parameter alpha, at
% parameter vector prvec 
%
% Inputs:
%  prvec [n x 1] - parameter vector (last element can be DC)
%  theta [2 x 1] - [rho; (precision)
%                   alpha (smoothness) ]
%  nx [1 x 1] - length of param vector to apply to prior (last element
%                  of prvec can be all-ones for dc term
%
% Outputs:
%         p [1 x 1] - log-prior
%        dp [n x 1] - grad
%   negCinv [n x n] - inverse covariance matrix (Hessian)
%    logdet [1 x 1] - log-determinant of -negCinv (optional)
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


if nargin <= 3
    DCflag = 0; % no DC dterm
    nx = length(prvec);
else
    DCflag = 1;
    if nargin < 5
	rhoDC = .1;
    end
    if length(prvec)~=(nx+1)
	error('mismatch in param vector size and nx');
    end
end


rho = max(precision,MINVAL);
const = rho/(1-aa.^2);

vdiag = [1;ones(nx-2,1)+aa^2;1]*const;
voffdiag = -ones(nx,1)*aa*const;
negCinv = -spdiags([voffdiag,vdiag,voffdiag],-1:1,nx,nx);

% Add prior indep prior variance on DC coeff
if DCflag
    negCinv = blkdiag(negCinv,-rhoDC);
end

dp = negCinv*prvec;
p = .5*sum(prvec.*dp,1);

if nargout>3
    logdetrm = nx*log(rho)-(nx-1)*log(1-aa.^2);
end
