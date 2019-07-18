function y = kronmulttrp(Amats,varargin)
% Multiply transpose of kronecker matrix by x
%
% y = kronmulttrp(Amats,x,ii);
% 
% Computes:
% y = (A_n kron A_{n-1} ... A_2 kron A_1)^T x
%
% See 'kronmult' for details

Amats = cellfun(@transpose,Amats,'UniformOutput',false);
y = kronmult(Amats,varargin{:});
