function M = mksparseshiftedbasis(waveform,ncols,delta)
% M = mksparseshiftedbasis(waveform,nrows,delta)
% 
%   Makes a sparse matrix with shifted copies of a waveform "waveform"
%
% INPUTS:
%
%  waveform [nw x 1] - column vector with waveform
%     ncols [1 x 1] - number of columns
%     delta [1 x 1] - shift between subsequent columns (must be integer)


nw = length(waveform);  % number of elements in waveform
nrows = nw+(ncols-1)*delta;  % number of total rows

ri = bsxfun(@plus,(1:nw)',0:delta:(ncols-1)*delta);  % row indices for non-zero elements
ci = repmat(1:ncols,nw,1);  % column indices for non-zero elements

M = sparse(ri,ci,repmat(waveform,1,ncols),nrows,ncols);  % Full matrix

