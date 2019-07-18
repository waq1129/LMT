function [xval,inds,yval] = argmax_pl(yy,varargin)
% [xval,inds,yval] = argmax(yy,varargin)
%
% Finds argmax of function: values of x1, x2, etc, at which yy achieves its maximum.
% 
% Inputs:
%
%  yy [m x n x p ...] - array of function values
%  x1 [m x 1] - coordinates along 1st dimension of yy
%  x2 [n x 1] - coordinates along 2nd dimension of yy (optional)
%  x3 [p x 1] - coordinates along 3rd dimension of yy (optional)
%  x4 [q x 1] - coordinates along 4th dimension of yy (optional)
%  
% OUTPUTS:
%  xval - vector of coordinates at which max is obtained
%  inds - indices at which xval is maximal
%  yval - value of function at max

ysize = size(yy);  % get dimensions of yy

% Check if it's a vector
if (ysize(1)==1) || (ysize(2) == 1)
    ysize = length(ysize);
end
    
[yval,inds] = max(yy(:)); % find maximum of yy

switch length(ysize)
    case 1,  % vector y
        xval = varargin{1}(inds);
    case 2, % matrix 
        [i1,i2] = ind2sub(ysize,inds); 
        xval = [varargin{1}(i1), varargin{2}(i2)]';
        inds = [i1, i2]';
    case 3,  % 3D array
        [i1,i2,i3] = ind2sub(ysize,inds);
        xval = [varargin{1}(i1), varargin{2}(i2), varargin{3}(i3)]';
        inds = [i1,i2,i3]';
    case 4, % 4D array
        [i1,i2,i3,i4] = ind2sub(ysize,inds);
        xval = [varargin{1}(i1), varargin{2}(i2), varargin{3}(i3), varargin{4}(i4)]';
        inds = [i1,i2,i3,i4]';

end
        
    

    