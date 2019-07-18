function [A] = covAR1iso(loghyper, x, z);

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
if nargin == 2
    z = x;
end

[n1, D] = size(x);
[n2, D] = size(z);

ell = exp(loghyper(1));                           % characteristic length scale
sf2 = exp(2*loghyper(2));                                     % signal variance


x1 = reshape(repmat(x',1,n2),[D,n1,n2]);
z1 = reshape(repmat(z',1,n1),[D,n2,n1]);
z1 = permute(z1,[1,3,2]);
dd = sum(abs(x1-z1)/ell,1);
dd = reshape(dd,[n1,n2]);

% compute test set covariances
A = sf2*exp(-dd);

