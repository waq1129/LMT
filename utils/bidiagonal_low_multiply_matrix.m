function AX1 = bidiagonal_low_multiply_matrix(dd,ss,X)

X1 = [zeros(1,size(X,2));X(1:end-1,:)];
ss1 = [0; ss(1:end-1)];
AX1 = bsxfun(@times,dd,X)+bsxfun(@times,ss1,X1);
