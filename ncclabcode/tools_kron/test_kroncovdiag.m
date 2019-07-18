% test_kroncovdiag
%
% Short script for illustrating kroncovdiag, which computes the diagonal of
% a matrix after left and right multiplying by a kronecker matrix B
%
% In practice, the function kroncovdiag.m is useful computing marginal
% variance of regression coefficients by computing the diagonal of the
% matrix B*Lcov*B', where Lcov is the posterior covariance in some basis
% (e.g., Fourier), and B is given by (for a D=3 example):
%    B = B3 \kron B2 \krob B1
%      or in Matlab:
%    B = kron(B3,kron(B2,B1));
%
% The function saves substantial memory and time over constructing the
% kronecker matrix B explicitly

% Set size of component matrices
n1 = 15; m1 = 12;  % Matrix 1 size
n2 = 16; m2 = 13; % Matrix 2 size
n3 = 17; m3 = 14; % Matrix 3 size
mm = m1*m2*m3;

% Make matrices
B1 = randn(n1,m1)/n1;
B2 = randn(n2,m2)/n2;
B3 = randn(n2,m3)/n3;

% Make random positive semi-def L matrix
Lcov = randn(mm);
Lcov = Lcov*Lcov';  % symmetrize

% Compute kronecker-matrix multiply, original method
tic;
Bmat = kron(B3,kron(B2,B1));
Cdiag1 = sum(Bmat.*(Bmat*Lcov),2);
toc;

% Compute kronecker-matrix multiply, use fast method
tic;
BB = {B1,B2,B3};
Cdiag2 = kroncovdiag(BB,Lcov);
toc;

% check that they agree
err = abs(max([Cdiag1-Cdiag2]))
