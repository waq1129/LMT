function zaso = encapsulateTrials_preload(varargin)
% zaso = encapsulateTrials_preload(X, Y, tbX, tbY)
% Makes zaso from trial based datasets, possibly with bases functions.
% When zaso is initialized, it loads all trials and makes a giant stimulus
% matrix. Operations are equivalent to encapsulateTrials.m but more memory
% intensive and less I/O and looping. This is almost same as NOT using zaso,
% except that the interface is same as zaso, so you can switch back and forth.
%
% Input:
%   X: {nTrials x 1} consistent sequence of (T(i) x dx) matrices (independent v)
%   Y: {nTrials x 1} consistent sequence of (T(i) x dy) matrices (dependent v)
%   tbX: (optional) temporal basis function for X (preserves the time length)
%   tbY: (optional) temporal basis function for Y
%
% tbX expands (T x dx) matrix to (T x nTotalFeatures) matrix by convolving
% a set of basis functions
%
% Output:
%   zaso: ZASO!
%
% Trials with different lengths are represented either as
%   1. full matrix and associated index mask (TODO)
%   2. cell array
% When using sparse matrix, subindexing can be expansive, therefore cell array
% is preferred.
%
% $Id$

zzaso = encapsulateTrials(varargin{:});

%% Loop over the trials and stack the resulting matrices
rawX = zeros(zzaso.N, zzaso.dimx);
rawY = zeros(zzaso.N, zzaso.dimy);
for kTrial = 1:zzaso.nTrials
    trialX = zzaso.X(kTrial);
    trialY = zzaso.Y(kTrial);
    timeIdx = zzaso.mask(kTrial,1):zzaso.mask(kTrial,2);
    rawX(timeIdx, :) = trialX;
    rawY(timeIdx, :) = trialY;
end

sparsityX = nnz(rawX) / numel(rawX); % proportion of non-zeros
if sparsityX < 0.3
    fprintf('Sparse X [%.3f]\n', sparsityX);
    rawX = sparse(rawX);
end

sparsityY = nnz(rawY) / numel(rawY);
if sparsityY < 0.3
    fprintf('Sparse Y [%.3f]\n', sparsityY);
    rawY = sparse(rawY);
end

zaso = encapsulateRaw(rawX, rawY, [], [], true);
zaso.trialZaso = zzaso;

end
