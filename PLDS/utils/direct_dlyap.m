function P = direct_dlyap(A,Q, opts)
% solves APA' + Q = P for P

% assumes Q to be symmetric and positive definite, resulting in the same
% holding for output matrix P. 

% WARNING: code creates a numel(A)-by-numel(A) matrix during runtime and 
%          solves a linear system of equations from this. May crash if
%          A is a very large matrix. 

% WARNING: code will not check if Q is sym. and p.d. and may return
%          wrong results if Q does not meet these assumptions

if size(A,1) ~= size(A,2)
    error('A is not a square matrix')
end
if size(Q,1) ~= size(Q,2)
    error('Q is not a square matrix')
end
if size(A,1) ~= size(Q,1)
    error('A and Q need to be of the same dimensionality')
end
if nargin < 3
  opts.SYM = true; opts.POSDEF = true; 
end

P = linsolve( kron(A,A) - eye(numel(A)), -Q(:));
P = reshape(P, [size(A,1), size(A,2)]);
%P = 0.5 * ( P + P'); % should not need this, and would not want to have
                      % this in case that Q is not symmetric. 

end