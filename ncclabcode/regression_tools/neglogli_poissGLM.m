function [L,dL,ddL] = neglogli_poissGLM(wts,zaso,fnlin)
% [L,dL,ddL] = neglogli_poissGLM(wts,X,Y)
%
% Compute negative log-likelihood of data under Poisson regression model,
% plus gradient and Hessian
%
% INPUT:
% wts [m x 1] - regression weights
%   X [N x m] - regressors
%   Y [N x 1] - output (binary vector of 1s and 0s).
%       fnlin - func handle for nonlinearity (must return f, df and ddf)
%
% OUTPUT:
%   L [1 x 1] - negative log-likelihood
%  dL [m x 1] - gradient
% ddL [m x m] - Hessian

m = numel(wts);
switch nargout
    case 1
	L = zasoFxysum(zaso, @(x,y) neglogli_poissGLM_sub(x, y, wts, fnlin, 1));
    case 2
	r = zasoFxysum(zaso, @(x,y) neglogli_poissGLM_sub(x, y, wts, fnlin, 2));
	L = r(1);
	dL = r(2);
    otherwise
	r = zasoFxysum(zaso, @(x,y) neglogli_poissGLM_sub(x, y, wts, fnlin, 3));
	L = r(1);
	dL = r(1 + (1:m));
	ddL = reshape(r(m+2:end), m, m);
end

end % neglogli_poissGLM

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function L = neglogli_poissGLM_sub(x, y, wts, fnlin, choice)

m = numel(wts);
xproj = x*wts;

switch choice
    case 1
	L = -y'*xproj + sum(fnlin(xproj)); % neg log-likelihood
    case 2
	[f,df] = fnlin(xproj); % evaluate nonlinearity
	L = zeros(m+1, 1);
	L(1) = -y'*xproj + sum(f); % neg log-likelihood
	L(1 + (1:m)) = x'*(df-y);         % gradient
    case 3
	[f,df,ddf] = fnlin(xproj); % evaluate nonlinearity
	L = zeros(m^2+m+1, 1);
	L(1) = -y'*xproj + sum(f); % neg log-likelihood
	L(1 + (1:m)) = x'*(df-y);         % gradient
	H = x'*bsxfun(@times,x,ddf); % Hessian
	L(m+2:end) = H(:);
end

end % neglogli_poissGLM_sub
