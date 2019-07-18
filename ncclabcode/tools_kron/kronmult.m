function y = kronmult(Amats,x,ii)
% Multiply kronecker matrix by matrix x
%
% y = kronmult(Amats,x,ii);
% 
% INPUT
%   Amats - cell array with matrices {A1, ..., An}
%       x - matrix to multiply with kronecker matrix formed from Amats
%      ii - binary vector indicating sparse locations of x rows (OPTIONAL)
%
% OUTPUT
%    y - vector output 
%
% Equivalent to (for 4th-order example)
%    y = (A4 \kron A3 \kron A2 \kron A1) * x
% or in matlab:
%    y = kron(A4,kron(A3,kron(A2,A1)))*x
%
% For two A's and vector x, equivalent to left and right matrix multiply:
%    y = A1 * reshape(x,m,n) * A2; % and reshape to a vector
%
% Computational cost: 
%    Given A1 [n x n] and A2 [m x m], and x a vector of length nm, 
%    standard implementation y = kron(A2,A1)*x costs O(n^2m^2)
%    whereas this algorithm costs O(nm(n+m))

ncols = size(x,2);

% Check if 'ii' indices passed in for inserting x into larger vector
if nargin > 2
    x0 = zeros(length(ii),ncols);
    x0(ii,:) = x;
    x = x0;
end
nrows = size(x,1);

% Number of matrices
nA = length(Amats);

if nA == 1
    % If only 1 matrix, standard matrix multiply
    y = Amats{1}*x;
else
    % Perform remaining matrix multiplies
    y = x; % initialize y with x
    for jj = 1:nA
        [ni,nj] = size(Amats{jj}); %
        y = Amats{jj}*reshape(y,nj,[]); % reshape & multiply
        y =  permute(reshape(y,ni,nrows/nj,[]),[2 1 3]); % send cols to 3rd dim & permute
        nrows = ni*nrows/nj; % update number of rows after matrix multiply
    end
    
    % reshape to column vector
    y = reshape(y,nrows,ncols);

end

