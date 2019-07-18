function B = repcell(A, m)
% B = repcell(A, m)
%  
% Places 'm' copies of matrix A in a row cell array
% (like repmat, but for cell arrays)
%
% Updated: 3 Feb 2014, (JW Pillow) 

[ht,wid] = size(A);
B = mat2cell(repmat(A,1,m),ht,wid*ones(1,m));
