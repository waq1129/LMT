function xfull = upsamp(x,ntimes)
% xfull = upsample(x,ntimes)
%
% Upsample each column to produce a matrix that is ntimes*length(x) long
% (i.e., each row of x is repeated "counts" times).

[slen,swid] = size(x);
xfull = zeros(slen*ntimes,swid);

ii = 1:ntimes:ntimes*slen;
for j = 1:ntimes
    xfull(ii+(j-1),:) = x;
end
