function [bases ttb bcenters] = makeGammaTemporalBasis(nT, tau, nBasis, verbose)
% [bases ttb bcenters] = makeGammaTemporalBasis(nT, tau, nBasis, verbose)
% Make Gamma temporal basis functions. This is good when there is a exponential
% decay in time in what you want to represent. The Gamma basis spans the same
% space as the Laguerre basis functions.
% If you want a smooth bases, just exclude the first basis (which is exponential)
%
% example usage:
%   nT = 3000; tau = 150; nBasis = 10;
%   makeGammaTemporalBasis(nT, tau, nBasis, true);
%
% Input
%   nT: number of time bins
%   tau: time constant
%   nBasis: number of basis vectors to return
%   verbose: (optional/default:false) plot the bases
%
% Output
%   bases: (nT x nBasis)
%
% CAUTION: if the thing you want to represent has a delay, Gamma basis will not
%   capture it very well.
%
% CAUTION: if the tau is too long compared to nT such that the bases run out
%   of (temporal) space, they will look wierd since they are normalized to have
%   L1 norm of 1.
%
% Requires 'filter' from signal processing toolbox.
%
% $Id$

if nargin < 4; verbose = false; end
assert(nT > 1); assert(tau > 0); assert(nBasis > 0);

a = [1 -exp(-1/tau)];
x = zeros(nT, 1);
x(1) = 1;

bases = zeros(nT, nBasis);
yy = filter(1, a, x);
bases(:, 1) = yy / sum(yy);
for k = 1:(nBasis-1)
    yy = filter(1, a, bases(:, k));
    bases(:, k+1) = yy / sum(yy);
end

if verbose; figure; plot(bases); title('Gamma basis functions'); end

if nargout > 1
    ttb = repmat((1:nT)', 1, nBasis);
    [C, bcenters] = max(bases);
end
