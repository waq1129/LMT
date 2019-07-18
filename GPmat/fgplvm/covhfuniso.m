function K = covhfuniso(loghyper, x, z)

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

if nargin==2
    z = x;
end

% K = sf2*exp(-sq_dist(x'/ell,-z'/ell)/2)./(sf2*exp(-sq_dist(x'/ell,-z'/ell)/2)+1)...
%     -sf2*exp(-sq_dist(x'/ell,z'/ell)/2)./(sf2*exp(-sq_dist(x'/ell,z'/ell)/2)+1);

K = sf2*exp(sq_dist(x'/ell,-z'/ell)/2)-sf2*exp(sq_dist(x'/ell,z'/ell)/2);
