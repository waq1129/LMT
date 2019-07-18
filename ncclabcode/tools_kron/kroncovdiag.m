function mdiag = kroncovdiag(BB,C,ii)
% Compute diagonal of BCB^T, where C is symmetric and B is kronecker matrix
%
% Ldiag = krondiagBXBtrp(B,C,ii)
% 
% Computes diagonal of B*C*B', where B = (Bn \kron ... \kron B2 \kron B1)
%
% INPUT
%      BB - cell array with basis matrices {A1, ..., An}
%       C - symmetric matrix
%      ii - binary vector indicating sparse locations of C rows (OPTIONAL)
%
% OUTPUT
%   mdiag - diagonal of matrix B*C*B'

% Compute some sizes
nB = length(BB); % Number of component matrices
nr = cellfun(@(x)size(x,1),BB); % number of rows for each component matrix
nc = cellfun(@(x)size(x,2),BB); % number of cols for each component matrix
nctot = prod(nc); % number of columns in B, and sidelength of C


% Expand C to size of B matrix, if necessary
if nargin > 2
    Cbig = zeros(nctot);
    Cbig(ii,ii) = C;
    C = Cbig;
end

% Compute mdiag 
if nB == 1
    mdiag = sum(BB{1}.*(BB{1}*C),2); % simple method for nB=1
else
    mdiag = C; % initialize mdiag with C
    for jj = 1:nB
        mdiag = BB{jj}*reshape(mdiag,nc(jj),[]); % 1st matrix multiply
        mdiag = reshape(mdiag,nr(jj),nctot/nc(jj),nc(jj),[]);  % reshape
        mdiag = permute(mdiag,[1 3 2 4]); % permute appropriately for 2nd matrix multiply
        mdiag = squeeze(sum(bsxfun(@times,mdiag,BB{jj}),2)); % 2nd matrix multiply
        mdiag = permute(mdiag,[2 3 1]); % permute to move these dimensions to back
        nctot = nctot/nc(jj); % reduce number of remaining columns
    end    
    mdiag = mdiag(:);
end
