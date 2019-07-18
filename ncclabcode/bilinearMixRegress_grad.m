function [w_hat,wt,wx,wlin] = bilinearMixRegress_grad(xx,xy,wDims,p,indsbilin,lambda,opts)
% [w_hat,wt,wx,wlin] = bilinearMixRegress_grad(xx,xy,wDims,p,indsbilin,lambda,opts)
% 
% Computes regression estimate with a bilinear parametrization of the
% parameter vector.  Uses grad and Hessian information to
% perform ascent.
%
% Finds solution to argmin_w ||y - x*w||^2 + lambda*||w||^2
% where w is parametrized as vec( wt*wx')
%
% Inputs:
%   xx - autocorrelation of design matrix (unnormalized)
%   xy - crosscorrelation between design matrix and dependent var (unnormalized)
%   wDims - [nt, nx] two-vector specifying # rows & # cols of w.
%   p - rank of bilinear filter
%   lambda - ridge parameter (optional)
%   opts - options struct (optional)
%
% Outputs:
%   wHat = estimate of full param vector
%   wt = column vectors
%   wx = row vectors
%
% $Id$

if (nargin >= 6) && ~isempty(lambda)  
    xx = xx + lambda*eye(size(xx)); % add ridge penalty to xx
end

if (nargin < 7) || isempty(opts)
    opts = optimset('gradobj', 'on', 'Hessian', 'on','display','iter');
else
    opts = optimset(opts, 'gradobj', 'on', 'Hessian', 'on');
end

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

% Initial value of w
w = [vec(wt*wx); wlin];
fval = .5*w'*xx*w - w'*xy;

% Initial parameter values
prs0 = [wt(:); wx(:); wlin(:)];
floss = @(prs)(bilinMixRegressLoss(prs,nt,nx,p,xx,xy,Ix,It,Ilin));
%HessCheck(floss,prs0+.1);
prs = fminunc(floss,prs0,opts);

% Place optimized parameters into correct form
wt = reshape(prs(1:nwt),nt,p);
wx = reshape(prs(nwt+1:nwt+nwx),p,nx);
wlin = prs(nwt+nwx+1:end);

% finally, put indices of w back into correct order
w_hat = zeros(nw,1);
w_hat(indsbilin) = vec(wt*wx);
w_hat(indslin) = wlin;



% ===========================================================
function [l,dl,H] = bilinMixRegressLoss(prs,nt,nx,p,C,b,Ix,It,Ilin)
%
% Computes .5*w'*C*w - w'b and its derivatives, where w is parametrized as
% w = vec(wt*wx), giving a bilinear (low-rank) parametrization of the
% matrix wt*wx.

% Unpack parameters
nnt = nt*p;
nnx = nx*p;
nwbi = nt*nx;
Wt = reshape(prs(1:nnt),nt,p);
Wx = reshape(prs(nnt+1:nnt+nnx),p,nx);
wlin = prs(nnt+nnx+1:end);
nlin = length(wlin);

wbi = vec(Wt*Wx);
w = [wbi; wlin];
Mt = kron(Ix,Wt);
Mx = kron(Wx',It);
Mtl = blkdiag(Mt,Ilin);
Mxl = blkdiag(Mx,Ilin);
Mxtrp = [Mx', sparse(nnt,nlin)];


% Loss function
l =  .5*w'*C*w - w'*b;

% Gradient
Cw_b = C*w-b;
dldt = Mxtrp*Cw_b;
dldx = Mtl'*Cw_b;
dl = [dldt;dldx];

% Hessian
Ht = Mxtrp*C*Mxtrp';  % upper left block (d^2 / dwt^2)
Hxl = Mtl'*C*Mtl;     % lower right block (d^2 /[dwx dwlin].^2 ) 
Hxlt = Mtl'*C*Mxtrp'; % first cross term t by xl (trivial part)
Hxlt2 = (vecpermcols(kron(reshape(Cw_b(1:nwbi),nt,nx)',eye(p)),nt,p)); % cross x by t terms (hard part)
Hxlt(1:nnx,:) = Hxlt(1:nnx,:)+Hxlt2;  % add x by t terms to relevant section
H = [Ht,Hxlt'; Hxlt, Hxl];  % assemble full Hessian

