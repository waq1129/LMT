function [fun,pp,Mspline,splinePrs]=fit_tspline_mse(x,y,ss)
% [[fun,pp,Mspline,splinePrs]=fit_tspline_mse(x,y,ss)
% 
%  Fit a function y = f(x) with a cubic spline, defined using a set of
%  breaks, smoothness and extrapolation criteria, by minimize MSE
%
% Inputs:
%   x,y - spline minimizes (f(x)-y).^2
%   ss - spline struct with fields:
%        .breaks - breaks between polynomial pieces
%        .smoothness - derivs of 1 less are continuous
%           (e.g., smoothness=3 -> 2nd derivs are continuous)
%        .extrapDeg - degree polynomial for extrapolation on ends
%        .tfun - transfer function (inverse link function)
%
% Outputs:
%   fun - function handle for nonlinearity (uses 'ppval')
%   pp - piecewise polynomial structure
%   Mspline - matrix for converting params to spline coeffs
%   splinePrs -  vector x such that spline coeffs = Mspline*x
%
% last updated: 26/03/2012 (JW Pillow)



% Compute design matrix
[Xdesign,Ysrt,Mspline] = mksplineDesignMat(x,y,ss,1);

nprs = size(Xdesign,2);
prs0 = randn(nprs,1)*.1;

% Set up loss function
floss = @(prs)loss_tspline_mse(prs,Xdesign,Ysrt,ss.tfun);
%HessCheck(floss,prs0);

% Solve least-squares regression problem
splinePrs = fminNewton(floss,prs0);  % use (simple implementation of) Newton's method

% % Use Matlab's fminunc instead:
% opts = optimset('display','on','gradobj','on','Hessian',
% 'off','maxiter',5000,'maxfunevals',5000,'largescale','off');
% splinePrs2 = fminunc(floss,prs0,opts);
% [floss(splinePrs) floss(splinePrs2)]

% Create function handle for resulting spline
[~,pp] = mksplinefun(ss.breaks, Mspline*splinePrs); 
fun = @(x)(ss.tfun(ppval(pp,x)));

% ========  MSE Loss function  =====================
function [L,dL,ddL] = loss_tspline_mse(prs,X,Y,g)
% [L,dL,ddL] = loss_tspline_mse(prs,X,Y,g);
%
% Compute MSE between f(Xdesign*prs) and Y

if nargout<=1
    % Compute MSE only
    L = sum(bsxfun(@plus,g(X*prs),-Y).^2,1);
elseif nargout == 2
    % Compute MSE & Gradient
    [f,df,] = g(X*prs);
    L = sum((f-Y).^2);
    dL = 2*X'*(df.*(f-Y));
else
    % Compute MSE, Gradient & Hessian
    [f,df,ddf] = g(X*prs);
    L = sum((f-Y).^2);
    dL = 2*X'*(df.*(f-Y));
    ddL = 2*X'*bsxfun(@times,X,df.^2+ddf.*(f-Y));
end