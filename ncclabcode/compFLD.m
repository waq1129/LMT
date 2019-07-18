function v = compFLD(A, B, rdge);
%  Computes Fisher linear discriminant for two sets of data 
%  v = compFLD(A, B, rdge);
%
%  Inputs:  
%    A,B - matrices 2 sets of data to discriminate each row contains
%          a single data point.  Must have the same # of cols
%    rdge - optional ridge parameter, for data too poorly sampled to fill
%           out the covariance of A and B.
%  Output:
%    v = unit vector which is the linear discriminant of A and B
%
%  Finds vector V that maximizes
%      v'[A; B]'[A;B]v /  (v'A'Av/m + v'B'Bv/b),
%  where m, n are the number of elements in A, B, respectively.
%  (i.e. the variance of the aggregated data projected onto v, divided
%   by the sum of variances of each data set projected onto v around its mean.

n = size(A,2);

if nargin < 3
    rdge = 0;
end
C = [A;B];  % aggregated data   


% rewrite problem so we're minimizing as y'M'My/(y'y), where y = (A + B)v
M = mysqrtm(cov(A) + cov(B));  % single value decomposition of denominator
Minv = inv(M + diag(ones(n,1)*rdge));

H = C*Minv; 

% Find eigenvect of M w/ biggest eigenval
opts.disp = 0; 
[y, ss] = eigs(cov(H)+diag(ones(n,1)*rdge), 1, 'lm', opts);

% now find v in terms of y.
v = Minv*y;  
if (mean(A)*v < 0)
    v = -v;  % Change orientation so A projects positively onto v
end
v = v./norm(v); %normalize to have unit length

% ----  
function Msqrt = mysqrtm(M);
% Msqrt = mysqrtm(M);
% 
% Function for computing the sqrt of symmetric matrices
% Handles ill-conditioned matrices gracefully
% Note: M must be symmetric!

[u,s,v] = svd(M);
Msqrt = u*sqrt(s)*u';

