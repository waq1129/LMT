function Z = multiconv(X,Y)
%  Z = multiconv(X,Y);
%  
%  Convolves each column (or appropriately sized chunk) of Y with X. Calls
%  sameconv to perform convolution of each set of Y columns with X
%
%  See also: sameconv

[nx,mx] = size(X);
[~,my] = size(Y);

if round(my/mx) ~= my/mx
    error('width of Y doesn''t divide width of X');
end

d = my/mx;
Z = zeros(nx,d);
for j = 1:d;
    Z(:,j) = sameconv(X,Y(:,(j-1)*mx+1:j*mx));
end
