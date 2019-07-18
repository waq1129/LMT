function [H, g] = compHess(fun, x0, dx, varargin)
% [H, g] = compHess(fun, x0, dx, varargin)
% Numerically computes the Hessian of a function fun around point x0
% expects fun to have sytax:  y = fun(x, varargin);
%
% Input:
%   fun: @(x) function handle of a real valued function that takes column vector
%   x0: (n x 1) point at which Hessian and gradient are estimated
%   dx: (1) or (n x 1) step size for finite difference
%   extra arguments are passed to the fun
%
% Output:
%   H: Hessian estimate
%   g: gradient estiamte

n = numel(x0);
H = zeros(n,n);
g = zeros(n,1);
f0 = feval(fun, x0, varargin{:});

% input check
if ~all(isfloat(dx) & isfinite(dx))
    error('dx must be finite and float');
end
if any(dx <= 0)
    error('dx must be strictly positive');
end

if isscalar(dx)
    vdx = dx*ones(n,1);
elseif numel(dx) == n
    vdx = dx(:);
else
    error('vector dx must be the same size as x0');
end
A = diag(vdx/2);

for j = 1:n  % compute diagonal terms
    %central differences
    f1 = feval(fun, x0+2*A(:,j),varargin{:});
    f2 = feval(fun, x0-2*A(:,j),varargin{:});
    H(j,j) = f1+f2-2*f0;
    g(j) = (f1-f2)/2;
    % forward differences
%     f1 = feval(fun, x0+2*A(:,j),varargin{:});
%     f2 = feval(fun, x0+A(:,j),varargin{:});
%     fx = feval(fun, x0,varargin{:});
%     H(j,j) = f1-2*f2+fx;
%     g(j)   = f2-fx;
end

for j = 1:n-1       % compute cross terms
    for i = j+1:n
        %central differences
        f11 = feval(fun, x0+A(:,j)+A(:,i),varargin{:});
        f22 = feval(fun, x0-A(:,j)-A(:,i),varargin{:});
        f12 = feval(fun, x0+A(:,j)-A(:,i),varargin{:});
        f21 = feval(fun, x0-A(:,j)+A(:,i),varargin{:});
        H(j,i) = f11+f22-f12-f21;
        % forward differences
%         fx = feval(fun, x0,varargin{:});
%         f1 = feval(fun, x0+A(:,j)+A(:,i),varargin{:});
%         f12 = feval(fun, x0+A(:,j),varargin{:});
%         f21 = feval(fun, x0+A(:,i),varargin{:});
%         H(j,i) = f1-f12-f21-fx;

        H(i,j) = H(j,i);
    end
end

H = H./(vdx * vdx');
g = g./vdx;
