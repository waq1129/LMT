% test_kronmult
%
% Short script for illustrating speed of 'kronmult' - function for performing
% matrix-tensor products, in lieu of using making full matrix 'kron'


% Size of component matrices
n1 = 23; m1 = 22;  % Matrix 1 size
n2 = 16; m2 = 18; % Matrix 2 size
n3 = 25; m3 = 25; % Matrix 3 size
mm = m1*m2*m3; % total number of colums (length of x)

% Make component matrices for kronecker product
A1 = randn(n1,m1)/sqrt(n1);
A2 = randn(n2,m2)/sqrt(n2);
A3 = randn(n3,m3)/sqrt(n3);

% Make x vectors
x = randn(mm,5);

% Kronecker-matrix multiply, original method
tic;
y1 = kron(A3,kron(A2,A1))*x;
toc;

% Compute using fast method
tic;
AA = {A1,A2,A3};
y2 = kronmult(AA,x);
toc;

% check that they agree
err = abs(max([y1-y2]))
