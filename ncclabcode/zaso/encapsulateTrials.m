function zaso = encapsulateTrials(X, Y, tbX, tbY)
% zaso = encapsulateTrials(X, Y, tbX, tbY)
% Makes zaso from trial based datasets, possibly with bases functions.
% Operations on the data are looped over trials.
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

zaso.nTrials = numel(X);
assert(zaso.nTrials == numel(Y), 'X and Y must have same # of trials');
trialLengths = cellfun(@(x)(size(x,1)), X);
zaso.N = sum(trialLengths);
assert(zaso.N > 0, 'At least one time bin required');
assert(zaso.N == sum(cellfun(@(x)(size(x,1)), Y)), 'X and Y must have the same # of time bins');
zaso.Xraw = X;
zaso.Yraw = Y;

zaso.mask = zeros(zaso.nTrials, 2);
zaso.mask(:, 2) = cumsum(trialLengths);
zaso.mask(:, 1) = [1; zaso.mask(1:end-1, 2)+1];

if nargin < 4; tbY = @(y) y; end
if nargin < 3; tbX = @(x) x; end
zaso.tbX = tbX;
zaso.tbY = tbY;
zaso.X = @(idx) zaso.tbX(zaso.Xraw{idx});
zaso.Y = @(idx) zaso.tbY(zaso.Yraw{idx});
zaso.dimx = size(tbX(X{1}), 2); % effective dimension
zaso.dimy = size(tbY(Y{1}), 2);

% save the absolute index for beginning of each trial
zaso.sub2idx = @(subidx) sub2idx(zaso.mask, subidx);
zaso.Nsub = zaso.nTrials; % used for subindexing

zaso.desc = sprintf('%s(%s)', mfilename, datestr(now, 30));
zzaso.Id = '$Id$';

zaso.fxsum = @(varargin) trialSum(1, varargin{:});
zaso.fysum = @(varargin) trialSum(2, varargin{:});
zaso.fxysum  = @(varargin) trialSum(3, varargin{:});
zaso.fx = @(varargin) trialCompute(1, varargin{:});
zaso.fy = @(varargin) trialCompute(2, varargin{:});
zaso.fxy  = @(varargin) trialCompute(3, varargin{:});
zaso.farray = @trialArray;

end % encapsulateTrials

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [sum_result, agg_result] = trialArray(zaso, fct_sum, fct_agg, zasoIdx)
    sum_result = cell(length(fct_sum), 1);
    for k = 1:length(fct_sum)
	temp = feval(fct_sum{k}, zaso.X(1), zaso.Y(1));
	sum_result{k} = zeros(size(temp));
    end

    agg_result = cell(length(fct_agg), 1);
    for k = 1:length(fct_agg)
	temp = feval(fct_agg{k}, zaso.X(1), zaso.Y(1));
	agg_result{k} = zeros(size(temp, 1), zaso.N);
    end

    if nargin < 4; zasoIdx = 1:zaso.nTrials; end
    if isempty(zasoIdx); return; end

    for kTrial = zasoIdx
	trialX = zaso.X(kTrial);
	trialY = zaso.Y(kTrial);
	timeIdx = zaso.mask(kTrial,1):zaso.mask(kTrial,2);
	for k = 1:length(fct_sum)
	    sum_result{k} = sum_result{k} + feval(fct_sum{k}, trialX, trialY);
	end
	for k = 1:length(fct_agg)
	    agg_result{k}(:, timeIdx) = feval(fct_agg{k}, trialX, trialY);
	end
    end
end % trialArray

function s = trialCompute(argMagic, zaso, fct, zasoIdx)
    switch argMagic
	case 1
	    trialX = zaso.X(1);
	    s = fct(trialX);
	case 2
	    trialY = zaso.Y(1);
	    s = fct(trialY);
	case 3
	    trialX = zaso.X(1);
	    trialY = zaso.Y(1);
	    s = fct(trialX, trialY);
    end % swtich
    assert(size(s, 2) == zaso.mask(1,2), 'function handle must return (d x #timeBinsWithinTrial)');
    s = zeros(size(s, 1), zaso.N);

    if nargin < 4; zasoIdx = 1:zaso.nTrials; end
    if isempty(zasoIdx); return; end

    for kTrial = zasoIdx
	timeIdx = zaso.mask(kTrial,1):zaso.mask(kTrial,2);
	switch argMagic
	    case 1
		trialX = zaso.X(kTrial);
		s(:, timeIdx) = fct(trialX);
	    case 2
		trialY = zaso.Y(kTrial);
		s(:, timeIdx) = fct(trialY);
	    case 3
		trialX = zaso.X(kTrial);
		trialY = zaso.Y(kTrial);
		s(:, timeIdx) = fct(trialX, trialY);
	end % swtich
    end % for
end % trialCompute

function s = trialSum(argMagic, zaso, fct, zasoIdx)
    switch argMagic
	case 1
	    trialX = zaso.X(1);
	    s = fct(trialX);
	case 2
	    trialY = zaso.Y(1);
	    s = fct(trialY);
	case 3
	    trialX = zaso.X(1);
	    trialY = zaso.Y(1);
	    s = fct(trialX, trialY);
    end % swtich
    s = zeros(size(s));

    if nargin < 4; zasoIdx = 1:zaso.nTrials; end
    if isempty(zasoIdx); return; end

    for kTrial = zasoIdx
	timeIdx = zaso.mask(kTrial,1):zaso.mask(kTrial,2);
	switch argMagic
	    case 1
		trialX = zaso.X(kTrial);
		s = s + fct(trialX);
	    case 2
		trialY = zaso.Y(kTrial);
		s = s + fct(trialY);
	    case 3
		trialX = zaso.X(kTrial);
		trialY = zaso.Y(kTrial);
		s = s + fct(trialX, trialY);
	end % swtich
    end % for
end % trialCompute

function idx = sub2idx(mask, subidx)
% convert subidx to absolute time index using masks
    idx = [];
    for k = 1:length(subidx)
	idx = [idx mask(subidx(k),1):mask(subidx(k),2)];
    end
end
