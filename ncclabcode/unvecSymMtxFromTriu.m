function M = unvecSymMtxFromTriu(v)
% M = unvecSymMtxFromTriu(v)
% 
% Takes entries in vector v, which are the upper triangular elements in a symmetric
% matrix, and forms the symmetric matrix
%
% INPUT:
%    v - vector of upper diagonal entries in symmetric matrix


n = floor(sqrt(2*length(v))); % sidelength of matrix
M = zeros(n); % initialize matrix

ri = triuinds(n); % get indices into matrix
M(ri) = v;
M = M+M'-diag(diag(M));
