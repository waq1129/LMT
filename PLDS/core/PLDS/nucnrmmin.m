function [Y,D] = nucnrmmin( S, opts )
% x = nucnrmmin( Y, opts )
%   Find the minimum of lambda*||A(Y-DH)||_* + gamma*||D||_1 + f(Y) 
%   where A(Y) is the operator that mean-centers Y (that is, we are looking 
%   for a low-dimensional *affine* space rather than a low dimensional
%   subspace, as a way of accounting for mean firing rates.) and H is a
%   matrix of spike history terms (ignored if k = 0). This focuses strictly 
%   on the Poisson case for simplicity. The algorithm follows the ADMM 
%   method introduced by Liu, Hansson and Vandenberghe.
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
%   center - if true, miniminze the nuclear norm of the mean-centered data.
%       if false, just do vanilla nuclear norm minimization (A(x) = x)
%   q - the number of history steps to fit (if zero, ignore history)
%   gamma - the weight on the \ell_1 term, if k > 0
%   verbose - verbosity level.
%       0 - print nothing
%       1 - print every outer loop of ADMM
%       2 - print every inner loop of Newton's method and ADMM
%
% David Pfau, 2012-2013

% set default values
rho     = 1.3;
eps_abs = 1e-6;
eps_rel = 1e-3;
maxIter = 250;
nlin    = 'logexp';
lambda  = 1;
center  = 1;
verbose = 1;
q       = 0;
gamma   = 1;
if nargin > 1
    for field = {'rho','eps_abs','eps_rel','maxIter','nlin','lambda','center','k','gamma','verbose'}
        if isfield(opts, field)
            eval([field{1} ' = opts.' field{1} ';'])
        end
    end
end

nz = logical(sum(S,2));
S = S(nz,:); % remove rows with no spikes
[N,T] = size(S);
T = T-q;
H = zeros(N*q,T); % history term
for i = 1:q
    H((i-1)*N+(1:N),:) = S(:,i:end-q+i-1);
end
S = S(:,q+1:end);

if center && strcmpi(nlin,'soft')
    % There's probably some alternative to the standard Newton step that
    % can take advantage of what we know about the nonlinearity, but I'll
    % save that for another day.
    error('Newton''s method is ill-conditioned when combining mean-centering with locally-flat nonlinearities!')
end
lambda = lambda * sqrt(N*T);
gamma  = gamma  * T/N;

switch nlin
    case 'exp'
        f = @exp;
    case 'soft'
        f   = @(x) exp(x).*(x<0) + (1+x).*(x>=0);
        df  = @(x) exp(x).*(x<0) + (x>=0);
        d2f = @(x) exp(x).*(x<0);
    case 'logexp'
        f   = @(x) log(1+exp(x));
        df  = @(x) 1./(1+exp(-x));
        d2f = @(x) exp(-x)./(1+exp(-x)).^2;
end

if center
    A = @(x) bsxfun(@minus,x,mean(x,2));
else
    A = @(x) x;
end
A_adj = A;

switch nlin  % crude ADMM initialization
    case 'exp'
        x = log(max(S,1));
    case {'soft','logexp'}
        x = max(S-1,0);
end

X = zeros(N,T);
Z = zeros(N,T);
D = zeros(N,N*q);
G = zeros(N,N*q); % lagrange multiplier for inner loop of ADMM, mostly used for initialization near optimal point

nr_p = Inf; nr_d = Inf;
e_p = 0;    e_d = 0;
iter = 0;
if q == 0
    fprintf('Iter:\t Nuc nrm:\t Loglik:\t Objective:\t dX:\t\t r_p:\t\t e_p:\t\t r_d:\t\t e_d:\n')
else
    fprintf('Iter:\t Nuc nrm:\t Loglik:\t ||D||_1:\t Objective:\t dX:\t\t r_p:\t\t e_p:\t\t r_d:\t\t e_d:\n')
end
while ( nr_p > e_p || nr_d > e_d ) && iter < maxIter % Outer loop of ADMM
    stopping = Inf; % stopping criterion
    x_old = x;
    if verbose == 2, fprintf('\tNewton:\t Obj\t\t Stopping\t\n'); end
    while stopping/norm(x,'fro') > 1e-6 % Outer loop of Newton's method
        switch nlin
            case 'exp'
                h =  exp( x ); % diagonal of Hessian
                g =  exp( x ) - S; % gradient
            otherwise
                h =  d2f( x ) - S .* ( d2f( x ) .* f( x ) - df( x ).^2 ) ./ f( x ).^2;
                g =  df ( x ) - S .* df( x ) ./ f( x );
                g(isnan(g)) = 0;
                h(isnan(h)) = 0;
        end
        
        grad = g + rho*A( x ) - A_adj( rho * X + rho * A(D*H) - Z ); % update this to include spike history term.
        dx = -inv_hess_mult(h,grad);
        x = x + dx;
        stopping = abs(grad(:)'*dx(:));
        if verbose == 2 % verbosity level
            fprintf('\t\t %1.2e \t %1.2e\n', obj(x), stopping)
        end
    end
    dx = norm(x_old-x,'fro')/norm(x,'fro');

    if q > 0 % run ADMM for LASSO
        [D,G] = admm_lasso_mat( H - mean(H,2)*ones(1,T), A_adj( A(x) - X + Z/rho ), gamma/rho, D, G, verbose == 2 );
    end

    Ax_ = A( x - D*H );
    
    if T > N
        [v,s,u] = svd( Ax_' + Z'/rho, 0 );
    else
        [u,s,v] = svd( Ax_ + Z/rho, 0);
    end
    X_ = u*max( s - eye(min(N,T))*lambda/rho, 0 )*v';
    
    Z_ = Z + rho * ( Ax_ - X_ );
    
    % compute residuals and thresholds
    r_p = Ax_ - X_;
    r_d = rho * A_adj( X - X_ );
    e_p = sqrt(N*T) * eps_abs + eps_rel * max( norm( Ax_, 'fro' ), norm( X_, 'fro' ) );
    e_d = sqrt(N*T) * eps_abs + eps_rel * norm( A_adj( Z ), 'fro' );
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
    
    fval = sum( sum( f( x ) - S .* log( f( x ) ) ) );
    nn = sum( svd( A( x ) ) );
    
    % print
    iter = iter + 1;
    if q == 0
        fprintf('%i\t %1.4e\t %1.4e\t %1.4e\t %1.4e\t %1.4e\t %1.4e\t %1.4e\t %1.4e\n', ...
            iter, nn, fval, lambda*nn+fval, dx, nr_p, e_p, nr_d, e_d);
    else
        fprintf('%i\t %1.4e\t %1.4e\t %1.4e\t %1.4e\t %1.4e\t %1.4e\t %1.4e\t %1.4e\t %1.4e\n', ...
            iter, nn, fval, sum(abs(D(:))), lambda*nn+gamma*sum(abs(D(:)))+fval, dx, nr_p, e_p, nr_d, e_d);
    end
end
Y = zeros(length(nz),T);
Y( nz,:) = x;
Y(~nz,:) = -Inf; % set firing rates to zero for rows with no data. Used to make sure the returned value is aligned with the input

    function y = inv_hess_mult(H,x)
        % For the Newton step of ADMM, we must multiply a vector by the inverse of
        % H + rho*A'*A, where H is the Hessian of the smooth penalty term (which in
        % our case is diagonal and therefore simple) and A is the linear operator
        % in the objective ||A(x)||_* + f(x). In our problem the linear operator is
        % mean-centering the matrix x, which is a symmetric and idempotent operator
        % that can be written out in matrix form for m-by-n matrices as:
        %
        % A = eye(m*n) - 1/n*kron(ones(n,1),eye(m))*kron(ones(n,1),eye(m))'
        %
        % and so (H + rho*A*A')^-1*x can be efficiently computed by taking
        % advantage of Woodbury's lemma.
        
        if center % update this to include spike history term
            Hi = 1./(H + rho);
            foo = (sum(Hi.*x,2)./(ones(N,1)-sum(Hi,2)*rho/T));
            y = Hi.*x + rho/T*Hi.*(foo*ones(1,T));
        else
            y = x./(H + rho);
        end
    end

    function y = obj(x)
        
        foo = S.*log(f(x));
        foo(isnan(foo)) = 0; % 0*log(0) = 0 by convention
        y = sum(sum(f(x) - foo)) + sum(sum(Z.*A(x-D*H))) + rho/2*norm(A(x-D*H)-X,'fro')^2;
        
    end
end