function [p,dp,Cinv,logdetrm] = logprior_AR1_2D(prvec,theta,nyx,rhoDC)
% [p,dp,Cinv,logdetrm] = logprior_AR1_2D(prvec,theta,nyx,rhoDC)
%
% Evaluate Gaussian AR1 log-prior at parameter vector prvec, which has the
% shape of a matrix of size nyx. 
%
% Inputs:
%  prvec [n x 1] - parameter vector
%  theta [2 x 1] - hyperparams: [rho; (precision); alpha (smoothness)]
%    nyx [2 x 1] - size of image: [ny, nx]
%  rhoDC [1 x 1] - precision prior for DC coeff (optional; requires length
%                  of parameter vector to be 1 greater than prod(nyx)
%
% Outputs:
%      p [1 x 1] - log-prior
%     dp [n x 1] - grad
%   Cinv [n x n] - inverse covariance matrix (Hessian)
% logdet [1 x 1] - log-determinant of -Cinv (optional)
%
% Inverse prior covariance matrix given by:
%  C^-1 = rho/(1-a^2) [ 1 -a
%                      -a 1+a^2 -a                       
%                        .   .   .
%                          -a 1+a^2 -a

MINVAL = 1e-6;

nprs = length(prvec);
nim = prod(nyx);
DCflag = 0;
if nprs>nim
    DCflag = 1;
    if nargin<4
	rhoDC = .1;
    end
end
rho = max(theta(1),MINVAL);
aa = min(theta(2),1-MINVAL);

% Column AR1 covariance 
vdiag = [1;ones(nyx(1)-2,1)+aa^2;1];
voffdiag = -ones(nyx(1),1)*aa;
Cinv1 = -spdiags([voffdiag,vdiag,voffdiag],-1:1,nyx(1),nyx(1));

% Row AR1 covariance
vdiag = [1;ones(nyx(2)-2,1)+aa^2;1];
voffdiag = -ones(nyx(2),1)*aa;
Cinv2 = -spdiags([voffdiag,vdiag,voffdiag],-1:1,nyx(2),nyx(2));

% Full inverse covariance matrix on linear weights
Cinv = -kron(Cinv2,Cinv1)*(rho/(1-aa^2)^2);

% Add prior indep prior variance on DC coeff
if DCflag
    Cinv = blkdiag(Cinv,-rhoDC);
end

dp = Cinv*prvec; % gradient 
p = .5*sum(prvec.*dp,1); % log-prior

if nargout>3
    logdetrm = nprs*log(rho)-(2*nprs-sum(nyx))*log(1-aa.^2);
end
