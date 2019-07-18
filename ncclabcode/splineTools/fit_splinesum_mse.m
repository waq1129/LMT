function [ff,splfuns] = fit_splinesum_mse(Y,X,ss)
% [ff,splfuns] = fit_splinesum_mse(Y,X,ss)
% 
% Fit a multivariate function f(x) with a sum of 1D splines:
%     Y = f1(X(:,1)) + f2(X(:,2)) + f3(X(:,3) + ...
% via least-squares regression.
%
% Inputs: Y - dependent variable (column vector)
%         X - indep variables (each column is a regressor)
%         ss - structure with fields: "breaks", "smoothness", "extrapDeg"
%            - use cell array if different params for each regressor
% 
% Outputs: ff - handle to sum-of-splines function 
%          splfuns - cell array of individual spline functions
%
% last updated: 7 Apr 2012 (JW Pillow)

nsplines = size(X,2);

% 1. Build one design matrix for fitting all splines
[Xdesign,Mspline,spstruct,nprs] = mksplineSumDesignMat(X,ss);

% 2. Solve for spline prs 
prs = Xdesign\Y;

% 4. Compute piecewise polynomial coeffs and create func handle
for ispl = 1:nsplines
    iprs = sum(nprs(1:ispl-1));
    splprs  = prs(iprs+[1:nprs(ispl)]);
    [splfuns{ispl}] = mksplinefun(spstruct{ispl}.breaks,Mspline{ispl}*splprs);
end

ff = @(x)splineSum(splfuns,x);

% ------------------------------
function y = splineSum(f,x)
% Evaluates a sum of functions 
%
%  Inputs:
%     f - cell array of function handles
%     x - column i is the input to the corresponding function f{i}
%
%  Output:
%     y = f{1}(x(:,1) + f{2}(x(:,2)) + .... + f{n}(x(:,n))
%
  nfuns = length(f);
  y = zeros(size(x,1),1);
  for j = 1:nfuns
      y = y+f{j}(x(:,j));
  end
  
      
