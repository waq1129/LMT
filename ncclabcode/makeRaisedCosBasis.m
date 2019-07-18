function [iht, ihbas, ihbasis, ihctrs] = makeRaisedCosBasis(nb, dt, endpoints, b, zflag)
% [iht, ihbas, ihbasis, ihctrs] = makeRaisedCosBasis(nh, hdt, endpoints, b, zflag);
%
% Make nonlinearly stretched basis consisting of raised cosines
% Inputs:  nb = # of basis vectors (alternatively a structure)
%          dt = time bin separation for representing basis
%          endpoints = 2-vector containg [1st_peak  last_peak], the peak 
%                  (i.e. center) of the last raised cosine basis vectors
%          b = offset for nonlinear stretching of x axis:  y = log(x+b) 
%              (larger b -> more nearly linear stretching)
%          zflag = flag for making (if = 1) finest-timescale basis
%                  vector constant below its peak
%
%  Outputs:  iht = time lattice on which basis is defined
%            ihbas = orthogonalized basis
%            ihbasis = basis itself
%            ihctrs  = centers of each basis function
%
%  Example call
%  [iht, ihbas, ihbasis] = makeRaisedCosBasis(10, .01, [0 10], .1);

if nargin < 3
    if nargin == 2, zflag = dt;
    else zflag = 0;  
    end
    ihprs = nb;
    nb = ihprs.nh;
    dt = ihprs.hdt;
    if isfield(ihprs, 'endpoints')
        endpoints = ihprs.endpoints;
    elseif isfield(ihprs, 'hspan')
        endpoints = ihprs.hspan;
    else error('missing field');
    end
    b = ihprs.b;
elseif (nargin == 4)
    zflag = 0;
end

% nonlinearity for stretching x axis (and its inverse)
nlin = @(x)(log(x+1e-20));
invnl = @(x)(exp(x)-1e-20);

if b <= 0
    error('b must be greater than 0');
end

if zflag == 2
    nb = nb-1;
end

yrnge = nlin(endpoints+b);  
db = diff(yrnge)/(nb-1);      % spacing between raised cosine peaks
ctrs = yrnge(1):db:yrnge(2);  % centers for basis vectors
mxt = invnl(yrnge(2)+2*db)-b; % maximum time bin
iht = [0:dt:mxt]';
nt = length(iht);        % number of points in iht
%ff = inline('(cos(max(-pi,min(pi,(x-c)*pi/dc/2)))+1)/2', 'x', 'c', 'dc');  % raised cosine basis vector
ff = @(x,c,dc)((cos(max(-pi,min(pi,(x-c)*pi/dc/2)))+1)/2);
ihbasis = ff(repmat(nlin(iht+b), 1, nb), repmat(ctrs, nt, 1), db);

if zflag == 1  % set first basis vector bins (before 1st peak) to 1
    ii = find(iht<=endpoints(1));
    ihbasis(ii,1) = 1;
elseif zflag == 2
    ii = find(iht<ihprs.absref);
    if isempty(ii)
        warning('makeRaisedCosBasis:emptyii', 'first basis vector is ZEROS');
    end
    ih0 = zeros(size(ihbasis,1),1);
    ih0(ii) = 1;
    ihbasis(ii,:) = 0;
    ihbasis = [ih0,ihbasis];
end
    
ihbas = orth(ihbasis);  % use orthogonalized basis
ihctrs = invnl(ctrs);