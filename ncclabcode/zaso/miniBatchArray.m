function [sum_result, agg_result] = miniBatchArray(zaso, fct_sum, fct_agg, zasoIdx)
% uses zaso.nMiniBatch to obtain a sequential segment of samples.

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

if nargin < 4; zasoIdx = 1:zaso.N; end
if isempty(zasoIdx); return; end

for t = 1:zaso.nMiniBatch:numel(zasoIdx)
    zIdx = t:min(numel(zasoIdx), t + zaso.nMiniBatch - 1);
    idx = zasoIdx(zIdx);

    miniX = zaso.X(idx);
    miniY = zaso.Y(idx);
    for k = 1:length(fct_agg)
	agg_result{k}(:, zIdx) = feval(fct_agg{k}, miniX, miniY);
    end
    for k = 1:length(fct_sum)
	sum_result{k} = sum_result{k} + feval(fct_sum{k}, miniX, miniY);
    end
end
