function g = gsmooth(y, sig)
% gsmooth - smooth vector or matrix by filtering with a Gaussian 
%
% g = gsmooth(x, sig);
%
% Inputs:
%     y [MxN] - matrix or vector (if matrix, operates along columns only)  
%   sig [1x1] - stdev of smoothing Gaussian 
%
% Output:
%     g [MxN] - smoothed signal
%
% Note: normalizes filter to have unit-norm 
%
% See also: bcsmooth

if (sig <= 0)  % Return original if no smoothing width 
    g = y;
else
    [len,wid] = size(y);
    if (len == 1)  % Flip to column vector
        y = y';
        flipped=1;
        len = wid;
    else
        flipped=0;
    end
    [~,nx] = size(y);
    
    nflt = max(min(len, sig*5),3);
    x = (-nflt:nflt)';
    gfilt = normpdf(x, 0, sig);
    gfilt = gfilt./norm(gfilt);
    
    g = zeros(size(y));
    o = conv(ones(size(y,1),1), gfilt, 'same');
    for j = 1:nx
        g(:,j) = conv(y(:,j), gfilt, 'same') ./ o;
    end
    
    if flipped
        g = g';
    end
end
