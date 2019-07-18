function [ri,ci] = trilinds(nn,k)
%  [ri,ci] = trilinds(nn)
%  [ri,ci] = trilinds(nn,k)
%
%  trilinds(nn) - extract row and column indices of lower triangular elements of a matrix
%  of size nn.
%
%  trilinds(nn,k) - use only indices of the kth diagonal and below (central diag = 0)

if nargin == 1
    [ri,ci] = find(triu(ones(nn)));
else
    [ri,ci] = find(triu(ones(nn),k));
end
