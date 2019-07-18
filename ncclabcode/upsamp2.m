function Imfull = upsamp2(Im,ntimes)
% xfull = upsamp2(Im,ntimes)
%
% Upsample each pixel by a factor of ntimes. 
% e.g., if Im is a 5x3 image, and ntimes=10, output is a 50x30 image.
%
% Main use is to make images that don't get anti-aliased in Preview.

% Place image inside an extra copy of the outer edge;
[slen,swid] = size(Im);
Im2 = [Im([1,1:end,end],1), [Im(1,:);Im;Im(end,:)], Im([1,1:end,end],end)];

[ii,jj] = meshgrid(0:swid+1,0:slen+1);

dx = 1./ntimes;
ihi = .5+dx/2:dx:swid+.5;
jhi = .5+dx/2:dx:slen+.5;

[iihi,jjhi] = meshgrid(ihi,jhi);

Imfull = interp2(ii,jj,Im2,iihi,jjhi, 'nearest');
