function Cprior = makeASDpriorcov(prs, ncoords, sqdists)
% Cprior = makeASDpriorcov(prs, ncoords)
%
% Makes prior covariance matrix for Automatic Smoothness Determination
% (ASD)  (Sahani & Linden 2003)
%
% Inputs:
%   prs = parameters [mu, sig1, sig2, sig3, ....]
%         p0 = scale parameter.  diagonal of cov is exp(-p0)
%         del1, del2 = falloff of cov in direction 1, 2, etc.
%   ncoords = [nx, ny, nz] number of coordinates in each direction
%   sqdists = squared distance vector for parameters (leave blank if
%         parameters live in a pixel grid.
%         ASD uses cartesian distance between cooordinates to build prior
%         covariance matrix. If parameters form a grid, makeASDpriorcov
%         will build a grid in pixel coordinates
%         sqdists is a cell array pf size (numel(nccords) x 1)
%         default is {1:nx; 1:ny; 1:nz}
%
% Output:
%   Cprior = exp(-p0 - .5*(xi-xj)^2/delx - .5*(yi-yj)^2/dely - ...)
%
% $Id$

p0 = prs(1);
dels = prs(2:end);
D = length(ncoords);
assert(length(dels) == D, 'number of parameters and dimension mismatch');
if length(dels)==1
    dels = repmat(dels,D,1);
end

if nargin < 3 || isempty(sqdists)
    sqdists = cell(D,1);
    for ii = 1:D
        xx = 1:ncoords(ii);
        sqdists{ii} = bsxfun(@minus,xx,xx').^2;
    end
end

for ii = 1:D
    assert(numel(sqdists{ii})==ncoords(ii)^2, 'dimensions must match coordinate')
end

switch D
    case 1, 
	Cprior = exp(-p0 - .5*sqdists{1}./dels);

    case 2,
	Cpriorx = exp(-.5*sqdists{1}./dels(1));

	Cpriory = exp(-.5*sqdists{2}./dels(2));

	Cprior = kron(Cpriory,Cpriorx)*exp(-p0);
	
    case 3,
	Cpriorx = exp(-.5*sqdists{1}./dels(1));

	Cpriory = exp(-.5*sqdists{2}./dels(2));

	Cpriorz = exp(-.5*sqdists{3}./dels(3));

	Cprior = kron(Cpriorz,kron(Cpriory,Cpriorx))*exp(-p0);

    otherwise
	error('makeASDpriorcov handles only 3-dimensions or fewer');
end
