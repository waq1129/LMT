function cnv = circonv(v1, v2)
%CIRCONV
%     circonv(v1, v2)
%     vector1 vector2
%  Purpose:  Convolve v1 with v2 as though both live on a circle. 
%            v1 is padded with zeros if smaller than v2

%  Note:  assumes first element of v1 is filter center if 

%  size(v1) = size(v2), otherwise it assumes v1 must be shifted

%  Assumes real


sz1 = length(v1);
sz2 = length(v2);

if (sz1 > sz2)

   error('Error --  v1 > v2  -- circonv');
elseif (sz1 < sz2)
      V1new = v2*0;
      V1new(find(v1~=nan)) = v1;
      V1new = shift(V1new, -floor(sz1/2));
      cnv = real(ifft(fft(V1new).*fft(v2))); 
else
      cnv = real(ifft(fft(v1).*fft(v2))); 
end

%%%--------------------------------------------------
function sMat = shift(x, q)
%%%  SHIFT(x, i)
%      shifts a matrix by q(1) places (vertically) and q(2) 
%      places (horizontally)
%    default is vertical shift if q = scalar.

[m, n] = size(x);
if length(q) < 2
   q(1) = q;
   if ((m == 1) | (n == 1))
      q(2) = q(1);
   else
      q(2) = 0;
   end
end

if (n == 1)
   sMat = shiftVert(x, -q(1));
elseif (m == 1)
   sMat = shiftVert(x', -q(2));
   sMat = sMat';
else
   sMat = shiftVert(x, -q(1));
   sMat = shiftVert(sMat', -q(2));
   sMat = sMat';
end

%%%-------------------------------------------------

function shX = shiftVert(xarg, sh)
% shifts vector 'xarg' by 'shft' places vertically, where
%  'sz' is the vertical length of xarg

[m, n] = size(xarg);
shX = xarg;

sh = mod(sh,m);
if (sh ~= 0)
   p1 = xarg(1:sh,:);
   p2 = xarg(sh+1:m,:);
   shX(1:m-sh,:) = p2;
   shX(m-sh+1:m,:) = p1;
end
   





