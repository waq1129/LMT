function plotRaster(sts, color, interval, isSorting, trialLines)
% plotRaster(sts, color, interval, isSorting);
% Draw a spike train raster plot
%
% Input:
%   sts: cell array of spike timings
%   color: (optional/black) color option for the lines
%   interval: (optional) plot only the spikes within [beginTime endTime)
%   isSorting: (optional/false) sort the spike trains by total number
%		 and first spike timing
%   trialLines: (optional) index for which trial the spikes came from to plot
%        the spikes on specific trial lines. defaults to 1:length(sts)
%
% Copyright 2006-2012 Memming, CNEL, all rights reserved

if ~iscell(sts)
    sts = {sts};
end

if nargin < 5
    trialLines = 1:length(sts);
end

if nargin < 2
    color = [0 0 0];
end

if nargin >= 3 && ~isempty(interval)
    timeStart = interval(1);
    timeEnd = interval(2);
    for k = 1:length(sts)
	st = sts{k};
	startIndex = find(st >= timeStart, 1, 'first');
	endIndex = find(st > timeEnd, 1, 'first') - 1;
        if isempty(endIndex)
            sts{k} = st(startIndex:end);
        else
            sts{k} = st(startIndex:endIndex);
        end
    end
end

if nargin < 4
    isSorting = false;
end

if isSorting
    % sort on number
    M = cellfun('length', sts);
    [sM, idx] = sort(M, 'ascend');
    sts = sts(idx);

    % TODO sort option
    emptyidx = cellfun('isempty', sts);
    firstAP = stcellfun(@(x)(x(1)), sts(~emptyidx));
    [dummy, sidx] = sort(firstAP);
    sts(~emptyidx) = sts(sum(emptyidx)+sidx);

    % sort on number
    M = cellfun('length', sts);
    [sM, idx] = sort(M, 'ascend');
    sts = sts(idx);
end

for k = 1:length(sts)
    st = sts{k}(:)';
    line([st; st], [trialLines(k); trialLines(k)+1]*ones(size(st)) + [0.1; -0.1]*ones(size(st)), 'Color', color);
end

set(gca, 'Box', 'on');
ylim([1, max(trialLines)+1]);
