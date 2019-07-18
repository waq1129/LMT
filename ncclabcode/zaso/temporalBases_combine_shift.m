function combinedW = temporalBases_combine_shift(w, basis, shifts, addDC)
% combinedW = temporalBasis_combine_shifts(w, bb, shifts, addDC);
% Weight the basis vectors by w and recover the set of kernels for each event.
%
% Input
%   basis: this gets shifted by shifts and multiplied by w
%
% See also: temporalBases_sparse_shift, temporalBases_combine
%
% $Id$

if nargin < 4; addDC = 0; end
if addDC; addDC = 1; end

w = w(:);
nEvents = size(shifts, 1);
combinedW = zeros(size(bb, 1), nEvents + addDC);

for kEvent = 1:nEvents
    bidx = basisIndices(kEvent, :);
    widx = sum(sum(basisIndices(1:(kEvent-1),:))) + (1:sum(bidx));
    combinedW(:, kEvent) = sum(bsxfun(@times, bb(:, bidx), w(widx)'), 2);
end

if addDC
    combinedW(:, end) = w(end);
end
