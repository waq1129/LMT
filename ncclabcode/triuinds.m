function [ri,ci] = triuinds(nn,k)
%  [ri,ci] = triuinds(nn)
%  [ri,ci] = triuinds(nn,k)
%
%  triuinds(nn) - extract row and column indices of upper triangular elements of a matrix
%  of size nn.
%
%  triuinds(nn,k) - use only indices of the kth diagonal and above (middle diag = 0)

if nargout == 1
    if nargin == 1
        ri = find(triu(ones(nn)));
    else
        ri = find(triu(ones(nn),k));
    end
    
else
    
    if nargin == 1
        [ri,ci] = find(triu(ones(nn)));
    else
        [ri,ci] = find(triu(ones(nn),k));
    end

end