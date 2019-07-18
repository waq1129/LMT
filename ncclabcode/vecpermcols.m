function [newA,inds] = vecpermcols(A,m,n)
% [newA,inds] = vecpermcols(A,m,n)
% 
% Permute the columns of A so that A*vec(x') is the same as
% vecpermcols(A,m,n)*vec(x), where x is an (m x n) matrix 

inds = vec(reshape((1:m*n),m,n)');
newA(:,inds) = A;
