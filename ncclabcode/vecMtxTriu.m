function v = vecMtxTriu(M,k)
% v = vecMtxTriu(M,k)
% 
% Take the upper triangular of a matrix and return it as a vector.
%
% INPUT:
%    M - matrix
%    k - which diagonal to use (optional; default = 0)

if (nargin == 1)
    ri = triuinds(size(M));
    v = M(ri);
else
    ri = triuinds(size(M),k);
    v = M(ri);
end
