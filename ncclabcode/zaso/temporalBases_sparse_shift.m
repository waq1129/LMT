function BX = temporalBases_sparse_shift(X, base, shifts, addDC, isCausal)
% BX = temporalBases_sparse_shift(X, base, shifts, addDC, isCausal);
% Computes the convolution of X with the selected base functions.
% It cannot do partial time indices. Use it for trial-based computation.
%
% Each row of X marks events, and with corresponding base selected by shifts,
% it is convolved to the feature matrix.
%
% Input:
%   X: (T x dx) sparse matrix of events through time (T bins, dx events)
%   base: (TB x 1) TB bins of a single basis that will be shifted in time
%   shifts (dx x M) (integer) time shift of the base to use for each event
%	    put NaN to exclude certain time shifts
%	    for negative shifts, only causal part is used
%   addDC: (boolean/default:false) if true, append a row of ones for DC (bias)
%   isCausal: (boolean/default:true) if false, allow acausal filtering
%	    negative shifts do not truncated the basis
%
% Output:
%   BX: (T x nTotalFeatuers) full matrix of features
%
% $Id$

% test code:
% temporalBases_sparse_shift(sparse([0 0 0 0 1 0 0 0 0 0 0;0 0 1 0 1 0 0 0 0 0 0]'), [3;2;1], [0 2 5;nan 0 1])

if ~issparse(X)
    warning('zaso:$Id$', 'This routine is not optimal for non-sparse matrices');
end

if nargin < 4; addDC = 0; end
if addDC; addDC = 1; end
if nargin < 5; isCausal = true; end

[T, dx] = size(X);
[TB, M2] = size(base);
assert(M2 == 1, 'a single basis is needed');
M = size(shifts, 2);
assert(dx == size(shifts, 1), '# of events shd equal the 1st dim of shifts');

featuresPerRow = sum(~isnan(shifts), 2); % (dx x 1) # of features per each event
nTotalFeatuers = sum(featuresPerRow);
featureRowIndex = [0; cumsum(featuresPerRow)];
BX = zeros(T, nTotalFeatuers + addDC);
if addDC; BX(:, end) = 1; end

for k = 1:dx % for each event type
    idx = find(X(:, k));
    featureIdx = featureRowIndex(k);
    if length(idx) < 10
	for kShift = 1:M % for each shift for the event
	    s = shifts(k, kShift);
	    if isnan(s); continue; end % ignore NaNs
	    featureIdx = featureIdx + 1;
	    for kk = 1:length(idx) % for each event occurrence
		if isCausal % causal
		    timeIdx = min(T,idx(kk)+max(s,0)):min(T,idx(kk)+TB-1+s);
		    timeBasisIdx = 1-min(s,0):min(TB, T-idx(kk)+1-s);
		else % acausal
		    timeIdx = min(T,max(idx(kk)+s,1)):min(T,idx(kk)+TB-1+s);
		    timeBasisIdx = 1-min(idx(kk)+s,0):min(TB, T-idx(kk)+1-s);
		end

		if isempty(timeBasisIdx); continue; end
		if X(idx(kk), k) == 1
		    BX(timeIdx, featureIdx) = BX(timeIdx, featureIdx) + ...
			base(timeBasisIdx);
		else
		    BX(timeIdx, featureIdx) = BX(timeIdx, featureIdx) + ...
			base(timeBasisIdx) * X(idx(kk), k);
		end
	    end % kk
	end % kShift
    else
	% do convolution instead of for loop on sum
	target = filter(base, 1, full(X(:, k)));
	for kShift = 1:M % for each shift for the event
	    s = shifts(k, kShift);
	    if isnan(s); continue; end % ignore NaNs
	    featureIdx = featureIdx + 1;
	    
	    target2 = target;

	    if isCausal && s < 0 % causal
		target2 = filter(base(-s+1:end), 1, full(X(:, k)));
	    end
	    timeIdx = max(1, s+1):T;
	    timeBasisIdx = 1:(T - max(1, s+1) + 1);

	    BX(timeIdx, featureIdx) = target2(timeBasisIdx);
	end
    end
end
