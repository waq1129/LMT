function [xsamphat,wtsaffine] = align_xtrue(xsamp,xtrue,align_flag)
% align xsamp with xtrue
if nargin<3
    align_flag = 1;
end
if align_flag
    nx = size(xtrue,1);
    wtsaffine = [ones(nx,1) xsamp]\xtrue;
    xsamphat = [ones(nx,1) xsamp]*wtsaffine;
else
    xsamphat = xsamp;
end
