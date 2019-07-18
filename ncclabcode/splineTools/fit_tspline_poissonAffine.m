function [fun,pp,neglogli,splPrs,Mspline]=fit_tspline_poissonAffine(x,y,ss,dtbin,prs0,aa,bb,opts)
% [fun,pp,neglogli,splPrs,Mspline] =
% fit_tspline_poissonAffine(x,y,ss,dtbin,prs0,aa,bb)
% 
%  Fit function f(x) that maximizes likelihood under:
%      y ~ Poiss(aa*f(x)+bb )
%  where f(x) is parametrized as a tspline (transformed cubic spline)
%
% Inputs:
%       x [Nx1] - input variable
%       y [Nx1] - output (count) variable
%   ss [struct] - spline struct with fields:
%        .breaks - breaks between polynomial pieces
%        .smoothness - derivs of 1 less are continuous
%           (e.g., smoothness=3 -> 2nd derivs are continuous)
%        .extrapDeg - degree polynomial for extrapolation on ends
%        .tfun - nonlinear transfer function (forcing positive outputs)
%   dtbin [1x1]- time bin size (OPTIONAL; assumed 1 otherwise)
%   prs0 [Mx1] - initial guess at spline params (OPTIONAL)
%     aa [Nx1] - rate from other filters, this nonlinearities (multiply)
%     bb [Nx1] - rate from other rank-1 components of model (add)
%   opts [1x1] - optimization options (for fminNewton)
% 
% Outputs:
%   fun - function handle for nonlinearity (uses 'ppval')
%   pp - piecewise polynomial structure
%   neglogli - negative loglikelihood at parameter estimate
%   splinePrs -  vector x such that spline coeffs = Mspline*x
%   Mspline - matrix for converting params to spline coeffs
%
% Last updated: 26/03/2012 (JW Pillow)

% Parse inputs
if nargin<4, dtbin=1; end
if nargin<5, prs0=[]; end
if nargin<6, aa = 1;  end
if nargin<7, bb = 0;  end
if nargin<8, opts=[];  end


% determine if we can resort the elements of x and y
if (length(aa)==1)&&(length(bb)==1), sortflag=1;  % allow sorted design matrix
else sortflag=0;  % keep design matrix in original temporal order (to be in register with aa)
end

% Compute design matrix
[Xdesign,Ysrt,Mspline] = mksplineDesignMat(x,y,ss,sortflag);
nprs = size(Xdesign,2);
if isempty(prs0)
    prs0 = log(exp(1)-1)*(Xdesign\ones(size(Ysrt)));
end

% Set up loss function
floss = @(prs)loss_tspline_poiss(prs,Xdesign,Ysrt,ss.tfun,dtbin,aa,bb);
% HessCheck(floss,prs0);

%% minimize negative log-likelihood using Newton's method
[splPrs,neglogli] = fminNewton(floss,prs0,opts);  % use (simple implementation of) Newton's method

% % % Use Matlab's fminunc instead:
% opts = optimset('display','off','gradobj','on','Hessian','off','maxiter',5000,'maxfunevals',5000,'largescale','off');
% splinePrs2 = fminunc(floss,prs0,opts);
% [floss(splinePrs) floss(splinePrs2)]  % Compare results, if desired

% Create function handle for resulting tspline
[~,pp] = mksplinefun(ss.breaks, Mspline*splPrs); 
splfun = @(x)ppfun(pp,x);
tfun = ss.tfun;
fun = @(x)tsplinefun(x,splfun,tfun);

if floss(prs0)<floss(splPrs)
    warning('what the heck');
    keyboard;
end

% ========  Poisson Neg Log Li (Loss function)  =====================
function [L,dL,ddL] = loss_tspline_poiss(prs,X,Y,g,dtbin,aa,bb)
% [L,dL,ddL] = loss_tspline_mse(prs,X,Y,g);
%
% Compute MSE between f(Xdesign*prs) and Y

% Project params onto design matrix
z = X*prs;
etol = 1e-100;

if nargout<=1
    % Compute neglogli
    f = bsxfun(@plus,bsxfun(@times,g(z),aa*dtbin),bb*dtbin);
    f(f<etol)=etol;
    L = -Y'*log(f) + sum(f);
elseif nargout == 2
    % Compute neglogli & Gradient
    [f,df] = g(z);
    f = bsxfun(@plus,bsxfun(@times,f,aa*dtbin),bb*dtbin);
    f(f<etol)=etol;
    L = -Y'*log(f) + sum(f);
    % grad
    df = bsxfun(@times,df,aa*dtbin);
    wts = (df-(Y.*df./f));
    dL = X'*wts;
elseif nargout == 3
    % Compute neglogli, Gradient & Hessian
    [f,df,ddf] = g(z);
    f = bsxfun(@plus,bsxfun(@times,f,aa*dtbin),bb*dtbin);
    f(f<etol)=etol;
    L = -Y'*log(f) + sum(f);
    % grad
    df = bsxfun(@times,df,aa*dtbin);
    wts = (df-(Y.*df./f));
    dL = X'*wts;
    % Hessian
    ddf = bsxfun(@times,ddf,aa*dtbin);    
    ww = ddf-Y.*(ddf./f-(df./f).^2);
    ddL = X'*bsxfun(@times,X,ww);
end
