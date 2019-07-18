function [w_hat,wt,wx,wlin] = bilinearMixRegress_coordAscent(xx,xy,wDims,p,indsbilin,lambda,opts)
% [w_hat,wt,wx,wlin] = bilinearMixRegress_coordAscent(xx,xy,wDims,p,indsbilin,lambda,opts)
% 
% Computes regression estimate with a bilinear parametrization of part of
% the parameter vector.
%
% Finds solution to argmin_w ||y - x*w||^2 + lambda*||w||^2
% where part of w is parametrized as vec(wt*wx')
%
% Inputs:
%   xx - autocorrelation of design matrix (unnormalized)
%   xy - crosscorrelation between design matrix and dependent var (unnormalized)
%   wDims - [nt, nx] two-vector specifying # rows & # cols of bilinearly parametrized w
%   p - rank of bilinear filter
%   indsbilin - indices to be parametrized bilinearly (the rest parametrize linearly)
%   lambda - ridge parameter (optional)
%   opts - options struct (optional)
%          fields: 'MaxIter' [25], 'TolFun' [1e-6], 'Display' ['iter'|'off']
%
% Outputs:
%   w  = estimate of full param vector
%   wlin = linearly parametrized portion
%   wt = column vectors (bilinear portion)
%   wx = row vectors (bilinear portion)
%
% $Id$

if (nargin >= 6) && ~isempty(lambda)  
    xx = xx + lambda*eye(size(xx)); % add ridge penalty to xx
end

if (nargin < 7) || isempty(opts)
    opts.default = true;
end

if ~isfield(opts, 'MaxIter'); opts.MaxIter = 25; end
if ~isfield(opts, 'TolFun'); opts.TolFun = 1e-6; end
if ~isfield(opts, 'Display'); opts.Display = 'iter'; end

% Set some params
nw = length(xy);
nbi = length(indsbilin);
nlin = nw-nbi;
nt = wDims(1);
nx = wDims(2);
nwt = p*nt;
nwx = p*nx;
It = speye(nt);
Ix = speye(nx);
Ilin = speye(nlin);

% Permute indices of XX and XY so that linear portion at beginning
indslin = setdiff(1:nw,indsbilin);  % linear coefficient indices
indsPrm = [indsbilin(:); indslin(:)]; % re-ordered indices
xx = xx(indsPrm,indsPrm);
xy = xy(indsPrm);
iibi = 1:nbi;     % new bilinear indices (first ones)
iilin = nbi+1:nw; % new linear indices (last ones)

% Initialize estimate of w by linear regression and SVD
w0 = xx\xy;
wlin = w0(iilin);
[wt,s,wx] = svd(reshape(w0(iibi),nt,nx));
wt = wt(:,1:p)*sqrt(s(1:p,1:p));
wx = sqrt(s(1:p,1:p))*wx(:,1:p)';

% Start coordinate ascent
w = [vec(wt*wx); wlin];
fval = .5*w'*xx*w - w'*xy;
fchange = inf;
iter = 1;
if strcmp(opts.Display, 'iter')
    fprintf('--- Coordinate descent of bilinear loss ---\n');
    fprintf('Iter 0: fval = %.4f\n',fval);
end

while (iter <= opts.MaxIter) && (fchange > opts.TolFun)
    
    % Update temporal components
    Mx = blkdiag(kron(wx',It),Ilin);
    wt = (Mx'*xx*Mx)\(Mx'*xy);
    wlin = wt(nwt+1:end);
    wt = reshape(wt(1:nwt), nt,p);
    
    % Update spatial components
    Mt = blkdiag(kron(Ix, wt),Ilin);
    wx = (Mt'*xx*Mt)\(Mt'*xy);
    wlin = wx(nwx+1:end);
    wx = reshape(wx(1:nwx),p,nx);

    % Compute size of change 
    w = [vec(wt*wx);wlin];
    fvalnew = .5*w'*xx*w - w'*xy;
    fchange = fval-fvalnew;
    fval = fvalnew;
    iter = iter+1;
    if strcmp(opts.Display, 'iter')
	fprintf('Iter %d: fval = %.4f,  fchange = %.4f\n',iter-1,fval,fchange);
    end
end

% finally, put indices of w back into correct order
w_hat = zeros(nw,1);
w_hat(indsbilin) = vec(wt*wx);
w_hat(indslin) = wlin;
