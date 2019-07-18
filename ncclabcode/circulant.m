function M = circulant(v)
% CIRCULANT - circulant matrix with vector v on first row
%
% M = circulant(v);

v = v(:);
M = toeplitz([v(1);flipud(v(2:end))],v);