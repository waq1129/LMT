function [eYs X P J] = kdr1d(x,y,N,h,varargin)
% [eYs X P] = condexp2d(x,y,N,h)
% 
% Compute kernel regression estimate of the
% conditional expectation, E[ y | x ] where
% x is a 1-dimensional vector of samples from.
%
% INPUT: 
%    x - nx1 matrix of predictors
%    y - nx1 vector of responses
%    N - (optional) scalar discretization; default=100
%        (default, N = 100)
%    h - (optional) bandwidths for each dimension
%        (default, bandwidth selected assuming Gaussian
%            distribution on x).
%        For vector h, cross-validates to find best single value of h. 
%
% mode - (optional) 
%            approx: (default) approximate kernel regression
%             exact: exact kernel regression 
%
% OUPUT: 
%  eYs - expected value matrix
%    X - discretization of predictor variables
%    P - KD-estimated probability distribution of x
% 
% TBD: 
%    * Error bars
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%   Evan Archer, 4/11/2011   %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    DO_EXACT = 0;
    if(nargin>4)
        if(strcmpi(varargin(1), 'exact'))
           DO_EXACT = 1;
        end
    end
    
    if(nargin<3 || isempty(N))
        N = 100;
    end

    J = [];
    if(nargin<4 || isempty(h) )
        h0 = 1.06*(1/length(x))^(1/5);
        h = std(x)*h0;        
    elseif(isvector(h))
        % now let's cross-validate!!!! 
        hd = h;
        if(DO_EXACT)
            J = cmpR(x,y,hd);
        else
            J = cmpFastR(x,y,hd,N);
        end
        [~,ii] = min(J);
        h = hd(ii);
    end

    [eYs X P] = cmpKDR(x,y,N,h,DO_EXACT);
    
end

function J = cmpR(x,y,hd)
    yhat = zeros(size(y));
    
    J = zeros(size(hd));
    for hdx = 1:length(hd)
        h = hd(hdx);
        ctf = 3*h;
        K0 = normpdf(0,0,h);       
        m = zeros(size(y));
        p = zeros(size(y));

        fprintf('%d of %d\r', hdx, length(hd))  
        for idx = 1:length(x) 
            delta_x = abs(x-x(idx));
            ii = find(delta_x < ctf);

            w = normpdf(x(ii), x(idx), h);
            p(idx) = sum(w);
            yhat(idx) = y(ii)'*w / (p(idx) + eps);
            m(idx) = 1-K0/(p(idx) + eps); 
        end        
        J(hdx) = var((y - yhat)./m);
        
        if(isnan(J(hdx)))
            error('Computed NAN in risk.')
        end
    end
end

function J = cmpFastR(x,y,hd,N)
    stdev = std(x);
    dx = range(x)/N;
    X = [0:(N-1) inf]'*dx+min(x);
    % domain for kernel
    X0 = -3*stdev:dx:3*stdev;
    
    [~, bin] = histc(x, X);

    % accumulate       
    eY0 = accumarray(bin, y,[], @sum);
    P0  = accumarray(bin, y,[], @length);

    J = zeros(size(hd));
    for hdx = 1:length(hd)
        h = hd(hdx);

        K0 = normpdf(0,0,h);       

        % compute kernel for h
        k = normpdf(X0,0,h); 
        k = k/sum( dx*k(:));
        
        % smooth
        eY = conv(eY0,k,'same');
        P = conv(P0,k,'same');
        % compute conditional expectation
        eYs = eY./(P + eps);
        
        m = (1 - K0./(P(bin) + eps));
        J(hdx) = var((y-eYs(bin))./m);
    end   
end

function [eYs X P] = cmpKDR(x,y,N,h,DO_EXACT)
        
    dx = range(x)/N;

    X = [0:(N-1) inf]'*dx+min(x);
    
    eYs = nan(N,1);
    P   = nan(N,1);
    
    if(DO_EXACT)
        for idx = 1:N % x-axis
            w = normpdf(x, X(idx), h);
            p = sum(w);
            eYs(idx) = y'*w / p;
            P(idx) = p;
        end        
    else
        [~, b1] = histc(x, X);
        
        % approximate kernel
        X0 = -3*h:dx:3*h;
        k = normpdf(X0,0,h); % we want h to correspond to the standard deviation
        k = k/sum(k*dx);

        % accumulate       
        eY = accumarray(b1, y,[], @sum);
        P = accumarray(b1, y,[], @length);
        
        % smooth
        eY = conv(eY,k,'same');
        P = conv(P,k,'same');

        % compute conditional expectation
        eYs = eY./(P+eps);

    end
    
    P = P/sum(P*dx);

    % get rid of inf
    X = X(1:end-1);

end
