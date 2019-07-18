function g = bcsmooth(y, halfwidth)
% g = bcsmooth(y, halfwidth);
% Smooth a vector or matrix by filtering with a symmetric boxcar
%
% Inputs:
%          y [MxN] - matrix or vector (if matrix, operates along columns only)  
%  halfwidth [1x1] - half width of the boxcar
%
% Output:
%     g [MxN] - smoothed signal
%
% See also: gsmooth

assert(halfwidth >= 0, 'Halfwidth must be non-negative');
assert(rem(halfwidth,1) == 0, 'Halfwidth must be an integer');

if halfwidth == 0
    g = y;
    return
end

[len,wid] = size(y);
if (len == 1)  % Flip to column vector
    y = y';
    isFlipped = true;
    len = wid;
else
    isFlipped = false;
end
[~,nx] = size(y);

filt = ones(halfwidth * 2 + 1, 1) / (halfwidth * 2 + 1);

g = zeros(size(y));
o = conv(ones(size(y,1),1), filt, 'same');
for j = 1:nx
    g(:,j) = conv(y(:,j), filt, 'same') ./ o;
end

if isFlipped
    g = g';
end
