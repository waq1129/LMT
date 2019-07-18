function  [D, OD] = sym_blk_tridiag_inv_v1(AA,BB,adx,bdx)
% Compute block tridiagonal terms of the inverse of a *symmetric* block
% tridiagonal matrix. 
% 
% Note: Could be generalized to non-symmetric matrices, but it is not currently implemented. 
% Note: Could be generalized to compute *all* blocks of the inverse, but
% it's not necessary for my current application and hence not implemented.
%
% Note: AA and BB could be combined into a single variable, but I think that might get confusing. 
% 
% Input: 
%   AA - (n x n x Ka) unique diagonal blocks 
%   BB - (n x n x Kb) unique off-diagonal blocks
%  adx - (T x 1) vector such that (i,i)th block of A is 
%                   A_{ii} = AA(:,:,adx(ii))
%  bdx - (T-1 x 1) vector such that (i,i+1) block of A is 
%                   A_{i,i+1} = AA(:,:,bdx(ii))
% 
% Output: 
%   D  - (n x n x T) diagonal blocks of the inverse
%  OD  - (n x n x T-1) off-diagonal blocks of the inverse
% 
% From: 
% Jain et al, 2006
% "Numerically Stable Algorithms for Inversion of Block Tridiagonal and Banded Matrices"
% (c) Evan Archer, 2014
%
    assert(numel(size(AA)) == 3, 'Always a 3d-array. For scalar blocks, make A of size [1 x 1 x Ka].')
    assert(length(adx) == length(bdx)+1, 'Should be one less upper diagonal term than diagonal term.')
    assert(length(size(AA)) == 3, 'Always expect the block index on the 3rd dimension');
    assert(size(AA,1) == size(BB,1) && size(AA,2) == size(BB,2) );
    assert(size(AA,1) == size(AA,2), 'Input matrix must be square.')
    
    % We only need R when our matrix is non-symmetric. for us, it always will be
    % R = zeros(runinfo.nStateDim, runinfo.nStateDim, T);
    BB = -BB; % we gotta make them the negative of the blocks
    T = numel(adx);
    n = size(AA,1);
    S = zeros(n, n, T-1);
    D = zeros(n, n, T); % diagonal  
    OD = zeros(n, n, T-1); % off diagonal
    III = eye(n);
    % R(:,:,1) = AA0\BB;
    S(:,:,end) = BB(:,:,bdx(end))/ AA(:,:,adx(end));
    for idx = (T-2):-1:1
    %    R(:,:,idx) = (AA - BB'*R(:,:,idx-1))\BB;
       S(:,:,idx) = BB(:,:,bdx(idx)) / (AA(:,:,adx(idx+1)) - S(:,:,idx+1)*BB(:,:,bdx(idx+1))');
    end
    % Compute diagonal and off-diagonal blocks
    D(:,:,1) = pinv(AA(:,:,adx(1)) - BB(:,:,bdx(1))*S(:,:,1)');
    OD(:,:,1) = S(:,:,1)'*D(:,:,1);
    for idx = 2:T-1
       D(:,:,idx) = (AA(:,:,adx(idx)) - BB(:,:,bdx(idx)) * S(:,:,idx)')\(III + BB(:,:,bdx(idx-1))'*D(:,:,idx-1)*S(:,:,idx-1)); 
       OD(:,:,idx) = S(:,:,idx)'*D(:,:,idx);
    end
    D(:,:,end) = AA(:,:,adx(end)) \ (III + BB(:,:,bdx(T-1))'*D(:,:,end-1)*S(:,:,end));
end