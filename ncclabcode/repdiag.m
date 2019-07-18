function M = repdiag(X,nreps);
% M = repdiag(X,nreps);
%
% Constructs a block-diagonal matrix using 'nreps' repeats of the matrix X.
% (Similar to blkdiag, but doesn't require passing X multiple times).
%
% Creates full (sparse) output if X is full (sparse)

A = repcell(sparse(X),nreps);
M = blkdiag(A{:});