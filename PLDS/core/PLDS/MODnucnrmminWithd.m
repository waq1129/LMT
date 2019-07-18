function [Y,Xu,Xs,Xv,d,cost] = MODnucnrmminWithd( S, opts, varargin )
%
% [Y,Xu,Xs,Xv,d,cost] = MODnucnrmminWithd( S, opts, varargin )
%
%  obj function is -log(S|x) + lambda||X||_* + Tr[Z'(x-X)] + rho/2||x-X||^2_F
%
% opts -
%   rho - dual gradient ascent rate for ADMM
%   eps_abs - absolute threshold for ADMM
%   eps_rel - relative threshold for ADMM
%   lambda - strength of the nuclear norm penalty. multiplied by the square
%       root of the number of elements in y to make sure it properly scales
%       as the size of the problem increases
%   maxIter - maximum number of iterations to run if the stopping threshold
%       is not reached
%   nlin - the nonlinearity to be used in the Poisson log likelihood:
%       exp - exponential
%       soft - exponential for negative x, linear otherwise
%       logexp - log(1+exp(x)), a smooth version of "soft"
%   verbose - verbosity level.
%       0 - print nothing
%       1 - print every outer loop of ADMM
%       2 - print every inner loop of Newton's method and ADMM
%
% David Pfau, 2012-2013
% minor modifications by Lars Buesing, 2014
%

Yinit = [];
dinit = [];
Yext  = [];

f   = [];
df  = [];
d2f = [];


assignopts(who,varargin);

% set default values
rho     = 1.3;
eps_abs = 1e-10;%1e-6;
eps_rel = 1e-5;%1e-3;
maxIter = 250;
nlin    = 'exp';
lambda  = 1;
verbose = 0;

if nargin > 1
    for field = {'rho','eps_abs','eps_rel','maxIter','nlin','lambda','verbose'}
        if isfield(opts, field)
            eval([field{1} ' = opts.' field{1} ';'])
        end
    end
end

[N,T] = size(S);
if isempty(Yext)
    Yext = zeros(size(S));
end

switch nlin
    case 'exp'
        f = @exp;
    case 'soft'
        f   = @(x) exp(x).*(x<0) + (1+x).*(x>=0);
        df  = @(x) exp(x).*(x<0) + (x>=0);
        d2f = @(x) exp(x).*(x<0);
    otherwise
        disp('NucNormMin: using user-defined link function')
end

if ~isempty(Yinit)
    x = Yinit;
else
    switch nlin  % crude ADMM initialization
        case 'exp'
            x = log(max(S,1))-Yext;
        otherwise
            x = max(S-1,0)-Yext;
    end
end

if ~isempty(dinit)
    d = dinit;
    if isempty(Yinit)
        x = zeros(size(S));
    end
else
    d = mean(x,2);
    x = bsxfun(@minus,x,d);
end

X = zeros(N,T);
Z = zeros(N,T);


nr_p = Inf; nr_d= Inf;
e_p = 0;    e_d = 0;
iter = 0;
if verbose>0;
    fprintf('Iter:\t Nuc nrm:\t Loglik:\t Objective:\t dX:\t\t r_p:\t\t e_p:\t\t r_d:\t\t e_d:\n')
end

while ( nr_p > e_p || nr_d > e_d ) && iter < maxIter % Outer loop of ADMM
    stopping = Inf; % stopping criterion
    x_old = x;
    if verbose == 2, fprintf('\tNewton:\t Obj\t\t Stopping\t\n'); end
    while stopping/norm(x,'fro') > 1e-6 % Outer loop of Newton's method
        xEval = bsxfun(@plus,x+Yext,d);
        switch nlin
            case 'exp'
                h =  exp( xEval ); % diagonal of Hessian
                g =  exp( xEval ) - S; % gradient
            otherwise
                h =  d2f( xEval ) - S .* ( d2f( xEval ) .* f( xEval ) - df( xEval ).^2 ) ./ f( xEval ).^2;
                g =  df ( xEval ) - S .* df( xEval ) ./ f( xEval );
                g(isnan(g)) = 0;
                h(isnan(h)) = 0;
        end
        grad = g + rho*(x-X) + Z;
        dx = -inv_hess_mult(h,grad);
        
        %!!! fix here
        gd = sum(g,2);%+lambda*d;
        hd = sum(h,2);%+lambda;
        dd = -gd./hd;
        dd(isnan(dd)) = 0;
        % upadate
        x = x + dx;
        d = d + dd;
        
        stopping = abs(grad(:)'*dx(:)+dd'*gd);
        if verbose == 2 % verbosity level
            fprintf('\t\t %1.2e \t %1.2e\n', obj(x), stopping)
        end
    end
    dx = norm(x_old-x,'fro')/norm(x,'fro');
    
    
    if T > N
        [v,s,u] = svd( x' + Z'/rho, 0 );
    else
        [u,s,v] = svd( x + Z/rho, 0);
    end
    
    Xs = max( s - eye(min(N,T))*lambda/rho, 0 );
    Xu = u;
    Xv = v;
    
    X_ = u*max( s - eye(min(N,T))*lambda/rho, 0 )*v';
    
    Z_ = Z + rho * ( x - X_ );
    
    % compute residuals and thresholds
    r_p = x - X_;
    r_d = rho * ( X - X_ );
    e_p = sqrt(N*T) * eps_abs + eps_rel * max( norm( x, 'fro' ), norm( X_, 'fro' ) );
    e_d = sqrt(N*T) * eps_abs + eps_rel * norm( Z , 'fro' );
    nr_p = norm( r_p, 'fro' );
    nr_d = norm( r_d, 'fro' );
    
    % heuristics to adjust dual gradient ascent rate to balance primal and
    % dual convergence. Seems to work pretty well: we almost always
    % converge in under 100 iterations.
    if nr_p > 10*nr_d
        rho = 2*rho;
    elseif nr_d > 10*nr_p
        rho = rho/2;
    end
    
    % update
    X = X_;
    Z = Z_;
    
    % !!! mod here, report stuff from low rank rates
    fval = sum( sum( f( bsxfun(@plus,X+Yext,d) ) - S .* log( f( bsxfun(@plus,X+Yext,d) ) ) ) );
    nn = sum( svd( X ) );
    
    cost.loglike   = -fval;
    cost.nucnorm   = nn;
    cost.total     = -cost.loglike+lambda*cost.nucnorm;
    
    % print
    iter = iter + 1;
    if verbose>0
        fprintf('%i\t %1.4e\t %1.4e\t %1.4e\t %1.4e\t %1.4e\t %1.4e\t %1.4e\t %1.4e\n',iter, nn, fval, lambda*nn+fval, dx, nr_p, e_p, nr_d, e_d);
    end
    
end

Y = X;   %!!! returns low rank rates
sqrtXs = diag(sqrt(diag(Xs)));
cost.penaltyC = norm(Xu*sqrtXs,'fro')^2;
cost.penaltyX = norm(Xv*sqrtXs,'fro')^2;


    function y = inv_hess_mult(H,x)
        y = x./(H + rho);
    end
    function y = obj(x)
        foo = S.*log(f( bsxfun(@plus,x+Yext,d) ));
        foo(isnan(foo)) = 0; % 0*log(0) = 0 by convention
        y = sum(sum(f( bsxfun(@plus,x+Yext,d)  ) - foo)) + sum(sum(Z.*x)) + rho/2*norm(x-X,'fro')^2;
    end

end