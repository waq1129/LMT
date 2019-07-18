function T = traceinv(M)
% T = traceinv(M)
%
% Compute the trace of the inverse of a symmetric, positive definite matrix
% M.  
%
% Uses Cholesky factorization of M to provide faster result that
% trace(inv(M)). NOTE: M must be symmetric (Hermitian) and positive
% definite.


T = sum(sum(inv(chol(M)).^2));