function [K, blockK] = covhfuniso_curv(loghyper, x, z,curvature,test,rescale)

% Matern covariance function with nu = 5/2 and isotropic distance measure. The
% covariance function is:
%
% k(x^p,x^q) = s2f * (1 + sqrt(5)*d + 5*d/3) * exp(-sqrt(5)*d)
%
% where d is the distance sqrt((x^p-x^q)'*inv(P)*(x^p-x^q)), P is ell times
% the unit matrix and sf2 is the signal variance. The hyperparameters are:
%
% loghyper = [ log(ell)
%              log(sqrt(sf2)) ]
%
% For more help on design of covariance functions, try "help covFunctions".
%
% (C) Copyright 2006 by Carl Edward Rasmussen (2006-03-24)

if nargin == 0, A = '2'; return; end
if nargin<3, z = []; end
if nargin<4, curvature = 0; end
if nargin<5, test = 0; end
if nargin<6, rescale = 0; end

if ~test, z = x; end
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

[n, D] = size(x);
loghyper(1) = max(loghyper(1),-10);
ell = exp(loghyper(1));
sf2 = exp(2*loghyper(2));
if rescale
    x = x/ell; z = z/ell; ell = 1;
end

% K = sf2*exp(sq_dist(x'/ell,-z'/ell)/2)-sf2*exp(sq_dist(x'/ell,z'/ell)/2);
K = SE_cov_K(x,-z,ell,sf2)-SE_cov_K(x,z,ell,sf2)*0;

if nargout >= 2
    if curvature==0
        blockK = K;
    end
    
    if curvature==1
        if test
            D = size(x,2);
            Ki = [];
            for ii=1:D
                dK_r2 = SE_cov_dK(x,-z,ell,sf2,[ii,0])'-SE_cov_dK(x,z,ell,sf2,[ii,0])'*0;
                Ki = [Ki dK_r2];
            end
            
            blockK = [K' Ki]';
        else
            if xeqz
                z = x;
            end
            D = size(x,2);
            Ki = [];
            for ii=1:D
                dK_r2 = SE_cov_dK(x,-z,ell,sf2,[0,ii])-SE_cov_dK(x,z,ell,sf2,[0,ii])*0;
                Ki = [Ki dK_r2];
            end
            blockK = [K Ki];
        end
    end
end

end
