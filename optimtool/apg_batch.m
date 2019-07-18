function [x, fval] = apg_batch(X_INIT, datastruct, grad_f, prox_h, fun, dim_x, opts)
% apg v0.1b (@author bodonoghue)
%
% Implements an Accelerated Proximal Gradient method
% (Nesterov 2007, Beck and Teboulle 2009)
%
% solves: min_x (f(x) + h(x)), x \in R^dim_x
%
% where f is smooth, convex and h is non-smooth, convex but simple
% in that we can easily evaluate the proximal operator of h
%
% returns solution and last-used step-size (the step-size is useful
% if you're solving a similar problem many times serially, you can
% initialize apg with the last use step-size
%
% this takes in two function handles:
% grad_f(v,opts) = df(v)/dv (gradient of f)
% prox_h(v,t,opts) = argmin_x (t*h(x) + 1/2 * norm(x-v)^2)
%                       where t is the step size at that iteration
% if h = 0, then use prox_h = [] or prox_h = @(x,t,opts)(x)
% put the necessary function data in opts fields
%
% implements something similar to TFOCS step-size adaptation (Becker, Candes and Grant 2010)
% and gradient-scheme adaptive restarting (O'Donoghue and Candes 2013)
%
% quits when norm(y(k) - x(k+1)) < EPS * max(1, norm(x(k+1))
%
% optional opts fields defined are below (with defaults)
% to use defaults simply call apg with opts = []
% X_INIT = zeros(dim_x,1); % initial starting point
USE_RESTART = true; % use adaptive restart scheme
MAX_ITERS = 1e3; % maximum iterations before termination
EPS = 1e-8; % tolerance for termination
ALPHA = 1.01; % step-size growth factor
BETA = 0.5; % step-size shrinkage factor
QUIET = false; % if false writes out information every 100 iters
GEN_PLOTS = false; % if true generates plots of norm of proximal gradient
USE_GRA = true; % if true uses UN-accelerated proximal gradient descent (typically slower)
STEP_SIZE = []; % starting step-size estimate, if not set then apg makes initial guess
FIXED_STEP_SIZE = false; % don't change step-size (forward or back tracking), uses initial
% step-size throughout, only useful if good
% STEP_SIZE set
rho = 1/4;

if (~isempty(opts))
    if isfield(opts,'X_INIT');X_INIT = opts.X_INIT;end
    if isfield(opts,'USE_RESTART');USE_RESTART = opts.USE_RESTART;end
    if isfield(opts,'MAX_ITERS');MAX_ITERS = opts.MAX_ITERS;end
    if isfield(opts,'EPS');EPS = opts.EPS;end
    if isfield(opts,'ALPHA');ALPHA = opts.ALPHA;end
    if isfield(opts,'BETA');BETA = opts.BETA;end
    if isfield(opts,'QUIET');QUIET = opts.QUIET;end
    if isfield(opts,'GEN_PLOTS');GEN_PLOTS = opts.GEN_PLOTS;end
    if isfield(opts,'USE_GRA');USE_GRA = opts.USE_GRA;end
    if isfield(opts,'STEP_SIZE');STEP_SIZE = opts.STEP_SIZE;end
    if isfield(opts,'FIXED_STEP_SIZE');FIXED_STEP_SIZE = opts.FIXED_STEP_SIZE;end
end

% if quiet don't generate plots
GEN_PLOTS = GEN_PLOTS & ~QUIET;

if (GEN_PLOTS); errs = zeros(MAX_ITERS,2);end


%% get batches
n = datastruct.n;
permid = randperm(n);
batch_size = min(opts.batch_size,n);
num_batches = floor(n/batch_size);
num_batches
% num_epochs = 1;
% numLeft = n-num_batches*batch_size;

err1 = inf;
fvals = 0;

x = X_INIT; y=x;
theta = 1;
exitflag = 0;
for k=1:MAX_ITERS
    for batch = 1:num_batches
        % datastruct for each batch
        if batch==num_batches
            batch_id = permid((batch-1)*batch_size+1:end);
        else
            batch_id = permid((batch-1)*batch_size+1:batch*batch_size);
        end
        x_batch = datastruct.x(batch_id,:);
        y_batch = datastruct.y(batch_id,:);
        datastruct_batch = datastruct;
        datastruct_batch.x = x_batch;
        datastruct_batch.y = y_batch;
        datastruct_batch.n = batch_size;
        
        g = grad_f(y,datastruct_batch);
        
        x_old = x;
        y_old = y;
        
        if (isempty(STEP_SIZE) || isnan(STEP_SIZE))  %% initialize t
            T = 10; dx = T*ones(dim_x,1); g_hat = nan;
            while any(isnan(g_hat))
                dx = dx/T;
                x_hat = x + dx;
                g_hat = grad_f(x_hat,datastruct_batch);
            end
            t0 = norm(x - x_hat)/norm(g - g_hat);
        else
            t0 = STEP_SIZE;
        end
        t = t0;
        fk = feval(fun,x_old,datastruct_batch);
        x = y - t*g;
        
        if ~isempty(prox_h)
            x = prox_h(x,t);
        end
        gt = (y-x)/t;
        fk1 = feval(fun,x,datastruct_batch);
        ff=fk1;
        iter = 1;
        while iter<1e3 && fk1 > fk-t*g'*gt*1e-4 % fk1 > fk-t*g'*gt+t/2*sum(gt.^2)
            t = t*rho;
            x = y - t*g;
            
            if ~isempty(prox_h)
                x = prox_h(x,t);
            end
            
            gt = (y-x)/t;
            
            fk1 = feval(fun,x,datastruct_batch);
            ff=[ff;fk1];
            iter = iter+1;
            
        end
        fnew = ff(end);
        
        err1 = norm(fvals(end)-fnew,1)/max(1,norm(fvals(end),1));
        %     err2 = norm(x-x_old,1)/max(1,norm(x_old,1));
        err2=inf;
        %  err1 = norm(fvals(end)-fnew,2)/max(1,norm(fvals(end),2));
        %     err2 = norm(x-x_old,2)/max(1,norm(x_old,2));
        
        fvals = [fvals; fnew];
        if (GEN_PLOTS);
            errs(k,1) = err1;
            %err2 = norm(x-x_old)/max(1,norm(x));
            %errs(k,2) = err2;
        end
        
        if (err1 < EPS || err2 < EPS)
            if (err1 < EPS && ~(err2 < EPS))
                display(['Terminate since tolf<' num2str(EPS)])
            end
            
            if (err2 < EPS && ~(err1 < EPS))
                display(['Terminate since tolx<' num2str(EPS)])
            end
            
            if (err1 < EPS && err2 < EPS)
                display(['Terminate since tolx<' num2str(EPS) ' and tolf<' num2str(EPS)])
            end
            exitflag = 1;
            break;
        end
        
        if(~USE_GRA)
            theta = 2/(1 + sqrt(1+4/(theta^2)));
        else
            theta = 1;
        end
        
        if (USE_RESTART && (y-x)'*(x-x_old)>0)
            x = x_old;
            y = x;
            theta = 1;
        else
            y = x + (1-theta)*(x-x_old);
        end
        
        g_old = g;
        
        if (~QUIET && mod(k,1)==0)
            fprintf('iter num %i, batch %i, fval: %1.2e, tolf: %1.2e, tolx: %1.2e, step-size: %1.2e\n',k,batch,fnew,err1,err2,t);
        end
    end
    if exitflag
        break;
    end
    
end
if (~QUIET)
    fprintf('iter num %i, batch %i, fval: %1.2e, tolf: %1.2e, tolx: %1.2e, step-size: %1.2e\n',k,batch,fnew,err1,err2,t);
    fprintf('Terminated\n');
end
if (GEN_PLOTS)
    errs = errs(1:k,:);
    figure();semilogy(errs(:,1));
    xlabel('iters');title('norm(tGk)')
    %figure();semilogy(errs(:,2));
    %xlabel('iters');title('norm(Dxk)')
end
fval = fvals(end);

end