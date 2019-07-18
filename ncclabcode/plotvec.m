function h = plotvec(vector,formatStr)
% h = plotvec(vector, formatStr)
%
% Plot a vector from 0 to a vecto

if nargin == 1
    h = quiver(0,0,vector(1),vector(2),1,'linewidth', '2');
else
    h = quiver(0,0,vector(1),vector(2),1,formatStr);
end    
