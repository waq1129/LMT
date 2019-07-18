function [A,dA] = covhfuniso_deriv(loghyper, x, z)

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

[n D] = size(x);
ell = exp(loghyper(1));                           % characteristic length scale
sf2 = exp(2*loghyper(2));                                     % signal variance

if nargin == 2
    z = x;
end

A1 = sf2*exp(-sq_dist(x'/ell,z'/ell)/2);

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
dA1 = repmat(A1,1,D).*dd/ell^2;

bb = reshape(dA1,[],D)';
cc = reshape(bb,[D,nx,nz]);
dd = permute(cc,[1,3,2]);
ee = reshape(dd,D,[]);
ff = reshape(ee,D*nz,[])';

dA1 = ff;

z = -z;
A2 = sf2*exp(-sq_dist(x'/ell,z'/ell)/2);

z1 = vec(z);
z2 = repmat(z1,1,nx)';

dd = x7-z2;
dA2 = repmat(A2,1,D).*dd/ell^2;

bb = reshape(dA2,[],D)';
cc = reshape(bb,[D,nx,nz]);
dd = permute(cc,[1,3,2]);
ee = reshape(dd,D,[]);
ff = reshape(ee,D*nz,[])';

dA2 = ff;

A = A2-A1;
dA = -dA2-dA1;

