function combinedW = temporalBases_combine(w, bb, basisIndices, addDC, isCausal)
% combinedW = temporalBasis_combine(w, bb, basisIndices, addDC, isCausal);
% Weight the basis vectors by w and recover the set of kernels for each event.
%
%   addDC: (boolean/default:false) if true, append a row of ones for DC (bias)
%   isCausal: (boolean/default:true) if false, allow acausal filtering
%	    negative shifts do not truncated the basis
%
% See also: temporalBases_sparse, temporalBases_sparse_shift
%
% $Id$

if nargin < 4; addDC = 0; end
if addDC; addDC = 1; end
if nargin < 5; isCausal = true; end
if ~isCausal, error('acausal reconstruction is not supported'); end

w = w(:);
nEvents = size(basisIndices, 1);
combinedW = zeros(size(bb, 1), nEvents + addDC);

assert(numel(w) == (sum(basisIndices(:)) + addDC));

for kEvent = 1:nEvents
    bidx = basisIndices(kEvent, :);
    widx = sum(sum(basisIndices(1:(kEvent-1),:))) + (1:sum(bidx));
    combinedW(:, kEvent) = sum(bsxfun(@times, bb(:, bidx), w(widx)'), 2);
end

if addDC
    combinedW(:, end) = w(end);
end
