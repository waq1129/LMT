function Y = colbasisconv(A, B, padding)
% colbasisconv - convolve each column of A with each column of B 
%
% Y = colbasisconv(A, B, paddingflag);
%   
%  Inputs:
%         A [NxM] - tall matrix
%         B [RxS] - short matrix (each column is a temporal filter)
%   padding [1x1] - (optional; default=1)
%                   1: pad A with zeros (height Y = height A).
%                   0: no padding: valid part of convolution only            
%
%  Output: 
%     Y [N x M*S] -  contains A filtered causally with each colum of B.  
%                    i.e., [A * B(:,1), A * B(:,2), ... A * B(:,n)]
%
%  Notes:
%  - Convolution performed using conv2
%  - B not flipped, so first row of Y is [A(1,:)*B(1,1), A(1,:)*B(1,2), ...]
%
% (updated: JW Pillow 14/03/2011)

if nargin < 3
    padding = 1;
end

[am, an] = size(A);
[bm, bn] = size(B);
ncols = an*bn;  

if padding
    A = [zeros(bm-1,an); A]; % Pad A with zeros
    Y = zeros(am,ncols);     % Allocate space for Y
else
    Y = zeros(am-bm+1,ncols);
end
B = flipud(B);  % Flip B 


% Perform convolution for each column of B;
for i = 1:bn
    jj = (i-1)*an+1:i*an;
    Y(:,jj) = conv2(A,B(:,i),'valid');
end
