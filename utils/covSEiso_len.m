function [A,dA] = covSEiso_len(loghyper, x, z, a, test)

% Squared Exponential covariance function with isotropic distance measure. The
% covariance function is parameterized as:
%
% k(x^p,x^q) = sf2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
%
% where the P matrix is ell^2 times the unit matrix and sf2 is the signal
% variance. The hyperparameters are:
%
% loghyper = [ log(ell)
%              log(sqrt(sf2)) ]
%
% For more help on design of covariance functions, try "help covFunctions".
%
% (C) Copyright 2006 by Carl Edward Rasmussen (2007-06-25)

if nargin == 0, A = '2'; return; end              % report number of parameters
if nargin == 2, z = x; end
if nargin<4
    a = 1;
    test = 1;
end
[n D] = size(x);
ell = exp(loghyper(1:end-1));                           % characteristic length scale
sf2 = exp(2*loghyper(end));                                     % signal variance

A = sf2*exp(-sq_dist(x'./repmat(vec(ell),1,size(x,1)),z'./repmat(vec(ell),1,size(z,1)))/2);

if nargout>1
    nx = size(x,1);
    nz = size(z,1);
    
    x1 = vec(x);
    x2 = repmat(x1,1,nz)';
    x3 = reshape(x2,[nz,nx,D]);
    x4 = permute(x3,[1,3,2]);
    x5 = reshape(x4,nz,[]);
    x6 = reshape(x5,1,[]);
    x7 = reshape(x6',D*nz,[])';
    
    z1 = vec(z);
    z2 = repmat(z1,1,nx)';
    
    dd = x7-z2;
    ell1 = vec(repmat(ell.^2,size(z,1),1))';
    dA = bsxfun(@times, repmat(A,1,D).*dd, 1./ell1);
    
    bb = reshape(dA,[],D)';
    cc = reshape(bb,[D,nx,nz]);
    dd = permute(cc,[1,3,2]);
    ee = reshape(dd,D,[]);
    ff = reshape(ee,D*nz,[])';
    
    dA = ff;
end

