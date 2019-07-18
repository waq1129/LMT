function [fun,splfuns,neglogli] = fit_tsplinesum_poisson(Y,X,ss,tfun,dtbin,prs0)
% [ff,splfuns,neglogli] = fit_tsplinesum_poisson(Y,X,ss)
% 
% Fit a multivariate function f(x) with a sum of 1D splines:
%     Y ~ Poiss[ g(f1(X(:,1)) + f2(X(:,2)) + f3(X(:,3) + ... ) ]
% via maximum likelihood, where g
%
% Inputs: Y - dependent variable of Poisson  (column vector)
%         X - indep variables (each column is a regressor)
%         ss - structure with fields: "breaks", "smoothness", "extrapDeg"
%            - use cell array if different params for each regressor
%         tfun - nonlinear transfer function to make output positive
% 
% Outputs: ff - handle to total tsplinesum function 
%          tsplfuns - cell array of individual spline functions
%
% last updated: 7 Apr 2012 (JW Pillow)



% 1. Parse inputs, initialize parameters if not passed in
if nargin<5
    dtbin = 1; % assumed binsize in which spikes observed 
end
nsplines = size(X,2); % number of splines to fit


% 2. Build one design matrix for fitting all splines
[Xdesign,Mspline,spstruct,nprs] = mksplineSumDesignMat(X,ss);
if nargin<6
    prs0 = randn(sum(nprs),1)*.1;
end


% 3. Set loss function
floss = @(prs)neglogli_tspline_poiss(prs,Xdesign,Y,tfun,dtbin);
% HessCheck(floss,prs0);  % Check that Hessian is correct

% minimize negative log-likelihood using Newton's method
[prs,neglogli] = fminNewton(floss,prs0);  


% 4. Insert fitted parameters into spline structures and create func handle
splfuns = cell(1,nsplines);  % cell array of piece-wise polynomials
for ispl = 1:nsplines
    iprs = sum(nprs(1:ispl-1));
    splprs  = prs(iprs+(1:nprs(ispl)));
    [~,pp] = mksplinefun(spstruct{ispl}.breaks,Mspline{ispl}*splprs);
    splfuns{ispl} = @(x)ppfun(pp,x);
end

fun = @(x)tsplinesumfun(x,splfuns,tfun);  % function handle
